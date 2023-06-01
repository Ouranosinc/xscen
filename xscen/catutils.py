"""Catalog creation and path building tools."""
import json
import logging
import operator as op
import os
import queue
import string
import threading
import warnings
from collections.abc import Mapping, Sequence
from copy import deepcopy
from fnmatch import fnmatch
from functools import partial, reduce
from multiprocessing import Pool
from pathlib import Path, PosixPath
from typing import Any, Optional, Union
from urllib.request import HTTPError, urlopen

import cftime
import netCDF4
import pandas as pd
import parse
import xarray
import xarray as xr
import yaml
from intake_esm import esm_datastore
from pandas import isna

from .catalog import COLUMNS, DataCatalog, generate_id
from .config import parse_config
from .io import copy_dataset, get_engine
from .utils import CV, date_parser, ensure_correct_time, standardize_periods  # noqa

logger = logging.getLogger(__name__)


__all__ = ["parse_directory", "parse_from_ds"]
# ## File finding and path parsing ## #


@parse.with_pattern(r"([^\_\/\\]*)", regex_group_count=1)
def _parse_word(text: str) -> str:
    r"""Parse helper to match strings with anything except / \ or _."""
    return text


@parse.with_pattern(r"([^\/\\]*)", regex_group_count=1)
def _parse_level(text: str) -> str:
    r"""Parse helper to match strings with anything except / or \."""
    return text


@parse.with_pattern(r"(([\d]{4,8}(\-[\d]{4,8})?)|fx)", regex_group_count=3)
def _parse_datebounds(text: str) -> tuple[str, str]:
    if "-" in text:
        return text.split("-")
    if text == "fx":
        return None, None
    return text, text


EXTRA_PARSE_TYPES = {
    "_": _parse_level,
    "no_": _parse_word,
    "datebounds": _parse_datebounds,
}


def _find_assets(
    root: os.PathLike, exts: set[str], lengths: set[int], dirglob: Optional[str] = None
):
    """Iterate over files in a directory, filtering according path depth and extensions."""
    for top, alldirs, files in os.walk(root):
        # Remove zarr subdirectories from next iteration
        dirs = deepcopy(alldirs)
        for dr in dirs:
            if dr.endswith(".zarr"):
                alldirs.remove(dr)

        if (os.path.relpath(top, root).count(os.path.sep) + 1) not in lengths:
            continue

        if dirglob is not None and not fnmatch(top, dirglob):
            continue

        if "zarr" in exts:
            for dr in deepcopy(dirs):
                if os.path.splitext(dr)[-1] == ".zarr":
                    yield os.path.join(top, dr)
        else:
            for file in files:
                if os.path.splitext(file)[-1] in exts:
                    yield os.path.join(top, file)


def _compile_pattern(pattern: str) -> parse.Parser:
    r"""Compile a parse pattern (if needed) for quicker evaluation.

    The `no_` default format spec is added where no format spec was given.
    The field prefix "?" is converted to "_" so the field name is a valid python variable name.
    """
    if isinstance(pattern, parse.Parser):
        return pattern

    parts = []
    for pre, field, fmt, _ in string.Formatter().parse(pattern):
        if not fmt:
            fmt = "no_"
        if field:
            if field.startswith("?"):
                field = "_" + field[1:]
            if field == "DATES":
                fmt = "datebounds"
            parts.extend([pre, "{", field, ":", fmt, "}"])
        else:
            parts.append(pre)
    return parse.compile("".join(parts), EXTRA_PARSE_TYPES)


def _name_parser(
    path: os.PathLike,
    root: os.PathLike,
    patterns: list[Union[str, parse.Parser]],
    read_from_file: Optional[Union[list[str], dict]] = None,
    attrs_map: Optional[dict] = None,
    xr_open_kwargs: Optional[dict] = None,
):
    """Extract metadata information from the file path.

    Parameters
    ----------
    path : str
        Full file path.
    root : str
        Root directory. Only the part of the path relative to this directory is checked against the patterns.
    patterns : list of str or parse.Parser
        List of patterns to try in `parse.parse`. See :py:func:`parse_directory` for the pattern specification.
    read_from_file : list of string or dict, optional
        If not None, passed directly to :py:func:`parse_from_ds` as `names`.
        If None (default), only the path is parsed, the file is not opened.
    attrs_map : dict, optional
        If `read_from_file` is not None, passed directly to :py:func:`parse_from_ds`.
    xr_open_kwargs : dict, optional
        If `read_from_file` is not None, passed directly to :py:func:`parse_from_ds`.

    Returns
    -------
    dict or None
        The metadata fields parsed from the path using the first matching pattern.
        If no pattern matched, None is returned.

    See Also
    --------
    parse.parse
    parse_directory
    parse_from_ds
    """
    abs_path = Path(path)
    path = abs_path.relative_to(root)
    xr_open_kwargs = xr_open_kwargs or {}

    d = {}
    for pattern in map(_compile_pattern, patterns):
        res = pattern.parse(str(path))
        if res:
            d = res.named
            break
    else:
        return None

    d["path"] = abs_path
    d["format"] = path.suffix[1:]

    if read_from_file:
        fromfile = parse_from_ds(
            abs_path, names=read_from_file, attrs_map=attrs_map, **xr_open_kwargs
        )
        d.update(fromfile)

    # files with a single year/month
    if "DATES" in d:
        d["date_start"], d["date_end"] = d.pop("DATES")

    if "date_end" not in d and "date_start" in d:
        d["date_end"] = d["date_start"]

    # strip to clean off lost spaces and line jumps
    # do not include wildcarded fields (? was transformed to _ in _compile_pattern)
    return {
        k: v.strip() if isinstance(v, str) else v
        for k, v in d.items()
        if not k.startswith("_")
    }


def _parse_dir(
    root: os.PathLike,
    patterns: list[str],
    dirglob: Optional[str] = None,
    check_perms: bool = True,
    read_from_file: Optional[Union[list[str], dict]] = None,
    attrs_map: Optional[dict] = None,
    xr_open_kwargs: Optional[dict] = None,
    progress: bool = False,
):
    """Iterate and parses files in a directory, filtering according to basic pattern properties and optional checks.

    Parameters
    ----------
    root: Pathlike
        Path to walk through.
    patterns: list of strings or compiled parsers
        Patterns that the files will be checked against.
        The extensions of the patterns are extracted and only paths with these are returned.
        Also, the depths of the patterns are calculated and only paths of this depth under the root are returned.
    dirglob: str
        A glob pattern. If given, only parent folders matching this pattern are walked through.
        This pattern can not include the asset's basename.
    check_perms: bool
        If True, only paths with reading permissions for the current user are returned.
    read_from_file : list of string or dict, optional
        If not None, passed directly to :py:func:`parse_from_ds` as `names`.
        If None (default), only the path is parsed, the file is not opened.
    attrs_map : dict, optional
        If `read_from_file` is not None, passed directly to :py:func:`parse_from_ds`.
    xr_open_kwargs : dict, optional
        If `read_from_file` is not None, passed directly to :py:func:`parse_from_ds`.
    progress: bool
        If True, the number of found files is printed to stdout.

    Return
    ------
    List of dictionaries
        Metadata parsed from each found asset.
    """
    lengths = {patt.count(os.path.sep) for patt in patterns}
    exts = {os.path.splitext(patt)[-1] for patt in patterns}
    comp_patterns = list(map(_compile_pattern, patterns))

    q_found = queue.Queue()
    q_checked = queue.Queue()
    parsed = []

    def check_worker():
        # Worker that processes the checks.
        while True:
            path = q_found.get()
            valid = True
            if check_perms:
                valid = valid and os.access(path, os.R_OK)
            if valid:
                q_checked.put(path)
            q_found.task_done()

    def parse_worker():
        # Worker that parses the paths
        while True:
            path = q_checked.get()
            try:
                d = _name_parser(
                    path,
                    root,
                    comp_patterns,
                    read_from_file=read_from_file,
                    attrs_map=attrs_map,
                    xr_open_kwargs=xr_open_kwargs,
                )
            except Exception as err:
                logger.error(f"Parsing file {path} failed with {err}.")
            else:
                if d is not None:
                    parsed.append(d)
                    # Print number of files but on round numbers to limit the calls to stdout for large collections
                    if progress and all(
                        [(progress < N or (progress % N == 0)) for N in [10, 100, 1000]]
                    ):
                        print(f"Found {len(parsed)} files", end="\r")
            q_checked.task_done()

    CW = threading.Thread(target=check_worker, daemon=True)
    CW.start()

    PW = threading.Thread(target=parse_worker, daemon=True)
    PW.start()

    for path in _find_assets(root, exts, lengths, dirglob):
        q_found.put(path)

    q_found.join()
    q_checked.join()
    return parsed


@parse_config
def parse_directory(
    directories: list,
    patterns: list,
    *,
    id_columns: list = None,
    read_from_file: Union[
        bool,
        Sequence[str],
        tuple[Sequence[str], Sequence[str]],
        Sequence[tuple[Sequence[str], Sequence[str]]],
    ] = False,
    homogenous_info: dict = None,
    cvs: Union[str, PosixPath, dict] = None,
    dirglob: Optional[str] = None,
    xr_open_kwargs: Mapping[str, Any] = None,
    only_official_columns: bool = True,
    progress: bool = False,
    parallel_dirs: Union[bool, int] = False,
) -> pd.DataFrame:
    r"""Parse files in a directory and return them as a pd.DataFrame.

    Parameters
    ----------
    directories : list
        List of directories to parse. The parse is recursive.
    patterns : list
        List of possible patterns to be used by :py:func:`parse.parse` to decode the file names. See Notes below.
    id_columns : list
        List of column names on which to base the dataset definition. Empty columns will be skipped.
        If None (default), it uses :py:data:`ID_COLUMNS`.
    read_from_file : boolean or set of strings or tuple of 2 sets of strings.
        If True, if some fields were not parsed from their path, files are opened and
        missing fields are parsed from their metadata, if found.
        If a sequence of column names, only those fields are parsed from the file, if missing.
        If False (default), files are never opened.
        If a tuple of 2 lists of strings, only the first file of groups defined by the
        first list of columns is read and the second list of columns is parsed from the
        file and applied to the whole group.
        It can also be a list of those tuples.
    homogenous_info : dict, optional
        Using the {column_name: description} format, information to apply to all files.
    cvs: str or PosixPath or dict, optional
        Dictionary with mapping from parsed term to preffered terms for each column.
        May have an additional "attributes" entry which maps from attribute names in the files to
        official column names. The attribute translation is done before the rest.
        In the "variable" entry, if a name is mapped to None (null), that variable will not be listed in the catalog.
        A term can map to another mapping from field name to values, so that a value on one column triggers the filling of other columns.
    dirglob : str, optional
        A glob pattern for path matching to accelerate the parsing of a directory tree if only a subtree is needed.
        Only folders matching the pattern are parsed to find datasets.
    xr_open_kwargs: dict
        If needed, arguments to send xr.open_dataset() when opening the file to read the attributes.
    only_official_columns: bool
        If True (default), this ensures the final catalog only has the columns defined in :py:data:`COLUMNS`. Other fields in the patterns will raise an error.
        If False, the columns are those used in the patterns and the homogenous info. In that case, the column order is not determined.
        Path, format and id are always present in the output.
    progress : bool
        If True, a counter is shown in stdout when finding files on disk.
        If parallel_dirs is not False nor 1, progress won't be of much help.
    parallel_dirs: bool or int
        If True, each directory is searched in parallel. If an int, it is the number of parallel searches.
        This should only be significantly useful if the directories are on different disks.

    Notes
    -----
    - Offical columns names are controlled and ordered by :py:data:`COLUMNS`:
        ["id", "type", "processing_level", "mip_era", "activity", "driving_institution", "driving_model", "institution",
         "source", "bias_adjust_institution", "bias_adjust_project","experiment", "member",
         "xrfreq", "frequency", "variable", "domain", "date_start", "date_end", "version"]
    - Not all column names have to be present, but "xrfreq" (obtainable through "frequency"), "variable",
        "date_start" and "processing_level" are necessary for a workable catalog.
    - 'patterns' should highlight the columns with braces.
        This acts like the reverse operation of `format()`. Fields will match alphanumeric parts of the path,
        excluding the "_", "/" and "\" characters. The "_" format spec will allow underscores.
        Field names prefixed by "?" will match normally, but will not be included in the output.
        See the documentation of :py:mod:`parse` for more format spec options.

        The "DATES" field is special as it will only match dates, either as a single date (YYYY, YYYYMM, YYYYMMDD)
        assigned to "{date_start}" (with "date_end" automatically inferred) or two dates of the same format as "{date_start}-{date_end}".

        Example: `"{source}/{?ignored project name}_{?:_}_{DATES}.nc"`
        Here, "source" will be the full folder name and it can't include underscores.
        The first section of the filename will be excluded from the output, it was given a name (ignore project name) to make the pattern readable.
        The last section of the filenames ("dates") will yield a "date_start" / "date_end" couple.
        All other sections in the middle will be ignored, as they match "{?:_}".

    Returns
    -------
    pd.DataFrame
        Parsed directory files
    """
    homogenous_info = homogenous_info or {}
    xr_open_kwargs = xr_open_kwargs or {}
    if only_official_columns:
        columns = set(COLUMNS) - homogenous_info.keys()
        pattern_fields = {
            f
            for f in set.union(
                *(set(patt.named_fields) for patt in map(_compile_pattern, patterns))
            )
            if not f.startswith("_")
        } - {"DATES"}
        unrecognized = pattern_fields - set(COLUMNS)
        if unrecognized:
            raise ValueError(
                f"Patterns include fields which are not recognized by xscen : {unrecognized}. "
                "If this is wanted, pass only_official_columns=False to remove the check."
            )

    read_file_groups = False  # Whether to read file per group or not.
    if not isinstance(read_from_file, bool) and not isinstance(read_from_file[0], str):
        # A tuple of 2 lists
        read_file_groups = True
        if isinstance(read_from_file[0][0], str):
            # only one grouping
            read_from_file = [read_from_file]
    elif read_from_file is True:
        # True but not a list of strings
        read_from_file = columns

    if cvs is not None:
        if not isinstance(cvs, dict):
            with open(cvs) as f:
                cvs = yaml.safe_load(f)
        attrs_map = cvs.pop("attributes", {})
    else:
        attrs_map = {}

    parse_kwargs = dict(
        patterns=patterns,
        dirglob=dirglob,
        read_from_file=read_from_file if not read_file_groups else None,
        attrs_map=attrs_map,
        xr_open_kwargs=xr_open_kwargs,
        progress=progress,
    )

    if parallel_dirs is True:
        parallel_dirs = len(directories)

    parsed = []
    if parallel_dirs > 1:
        with Pool(processes=parallel_dirs) as pool:
            results = []
            for directory in directories:
                results.append(pool.apply_async(_parse_dir, (directory,), parse_kwargs))
            for res in results:
                parsed.extend(res.get())
    else:
        for directory in directories:
            parsed.extend(_parse_dir(directory, **parse_kwargs))

    if not parsed:
        raise ValueError("No files found.")
    else:
        if progress:
            print()
        logger.info(f"Found and parsed {len(parsed)} files.")

    # Path has become NaN when some paths didn't fit any passed pattern
    df = pd.DataFrame(parsed).dropna(axis=0, subset=["path"])

    if only_official_columns:  # Add the missing official columns
        for col in set(COLUMNS) - set(df.columns):
            df[col] = None

    # Parse attributes from one file per group
    def read_first_file(grp, cols):
        fromfile = parse_from_ds(grp.path.iloc[0], cols, attrs_map, **xr_open_kwargs)

        logger.info(f"Got {len(fromfile)} fields, applying to {len(grp)} entries.")
        out = grp.copy()
        for col, val in fromfile.items():
            for i in grp.index:  # If val is an iterable we can't use loc.
                out.at[i, col] = val
        return out

    if read_file_groups:
        for group_cols, parse_cols in read_from_file:
            df = (
                df.groupby(group_cols)
                .apply(read_first_file, cols=parse_cols)
                .reset_index(drop=True)
            )

    # Add homogeous info
    for key, val in homogenous_info.items():
        df[key] = val

    # Replace entries by definitions found in CV
    if cvs:
        for i in df.index:
            for col, reps in cvs.items():
                if col == "variable":
                    # Variable can be a tuple, we still want to replace individual names through the cvs
                    for oldvar, new in reps.items():
                        if oldvar in df.loc[i, col]:
                            if isinstance(new, dict):
                                for name, newval in new.items():
                                    if name == "variable":
                                        df.at[i, name] = [
                                            newval if v == oldvar else v
                                            for v in df.loc[i, name]
                                        ]
                                    else:
                                        df.at[i, name] = newval
                            else:
                                df.at[i, name] = [
                                    new if v == oldvar else v for v in df.loc[i, name]
                                ]
                else:
                    for oldval, new in reps.items():
                        if oldval == df.loc[i, col]:
                            if isinstance(new, dict):
                                for name, newval in new.items():
                                    if name == "variable":
                                        df.at[i, name] = [newval]
                                    else:
                                        df.at[i, name] = newval
                            else:
                                df.at[i, name] = new

    # translate xrfreq into frequencies and vice-versa
    if {"xrfreq", "frequency"}.issubset(df.columns):
        df["xrfreq"].fillna(
            df["frequency"].apply(CV.frequency_to_xrfreq, default=pd.NA), inplace=True
        )
        df["frequency"].fillna(
            df["xrfreq"].apply(CV.xrfreq_to_frequency, default=pd.NA), inplace=True
        )

    # Parse dates
    if "date_start" in df.columns:
        df["date_start"] = df["date_start"].apply(date_parser)
    if "date_end" in df.columns:
        df["date_end"] = df["date_end"].apply(date_parser, end_of_period=True)

    # Checks
    if {"date_start", "date_end", "xrfreq", "frequency"}.issubset(df.columns):
        # All NaN dates correspond to a fx frequency.
        invalid = (
            df.date_start.isnull()
            & df.date_end.isnull()
            & (df.xrfreq != "fx")
            & (df.frequency != "fx")
        )
        n = invalid.sum()
        if n > 0:
            warnings.warn(
                f"{n} invalid entries where the start and end dates are Null but the frequency is not 'fx'."
            )
            logger.debug(f"Paths: {df.path[invalid].values}")
            df = df[~invalid]

    # todo
    # - Vocabulary check on xrfreq and other columns
    # - Format is understood

    # Create id from user specifications
    df["id"] = generate_id(df, id_columns)

    # ensure path is a string
    df["path"] = df.path.apply(str)

    # Sort columns and return
    if only_official_columns:
        return df.loc[:, COLUMNS]
    return df


def parse_from_ds(
    obj: Union[os.PathLike, xr.Dataset],
    names: Sequence[str],
    attrs_map: Optional[Mapping[str, str]] = None,
    **xrkwargs,
):
    """Parse a list of catalog fields from the file/dataset itself.

    If passed a path, this opens the file.

    Infers the variable from the variables.
    Infers xrfreq, frequency, date_start and date_end from the time coordinate if present.
    Infers other attributes from the coordinates or the global attributes. Attributes names
    can be translated using the `attrs_map` mapping (from file attribute name to name in `names`).

    If the obj is the path to a Zarr dataset and none of "frequency", "xrfreq", "date_start" or "date_end"
    are requested, :py:func:`parse_from_zarr` is used instead of opening the file.
    """
    get_time = bool(
        {"frequency", "xrfreq", "date_start", "date_end"}.intersection(names)
    )
    if not isinstance(obj, xr.Dataset):
        obj = Path(obj)

    if isinstance(obj, Path) and obj.suffixes[-1] == ".zarr" and not get_time:
        logger.info(f"Parsing attributes from Zarr {obj}.")
        ds_attrs, variables = _parse_from_zarr(obj, get_vars="variable" in names)
        time = None
    elif isinstance(obj, Path) and obj.suffixes[-1] == ".nc":
        logger.info(f"Parsing attributes with netCDF4 from {obj}.")
        ds_attrs, variables, time = _parse_from_nc(
            obj, get_vars="variable" in names, get_time=get_time
        )
    else:
        if isinstance(obj, Path):
            logger.info(f"Parsing attributes with xarray from {obj}.")
            obj = xr.open_dataset(obj, engine=get_engine(obj), **xrkwargs)
        ds_attrs = obj.attrs
        time = obj.indexes["time"] if "time" in obj else None
        variables = set(obj.data_vars.keys()).difference(
            [v for v in obj.data_vars if len(obj[v].dims) == 0]
        )

    rev_attrs_map = {v: k for k, v in (attrs_map or {}).items()}
    attrs = {}

    for name in names:
        if name == "variable":
            attrs["variable"] = tuple(sorted(variables))
        elif name in ("frequency", "xrfreq") and time is not None and time.size > 3:
            # round to the minute to catch floating point imprecision
            freq = xr.infer_freq(time.round("T"))
            if freq:
                if "xrfreq" in names:
                    attrs["xrfreq"] = freq
                if "frequency" in names:
                    attrs["frequency"] = CV.xrfreq_to_frequency(freq)
            else:
                warnings.warn(
                    f"Couldn't infer frequency of dataset {obj if not isinstance(obj, xr.Dataset) else ''}"
                )
        elif name in ("frequency", "xrfreq") and time is None:
            attrs[name] = "fx"
        elif name == "date_start" and time is not None:
            attrs["date_start"] = time[0]
        elif name == "date_end" and time is not None:
            attrs["date_end"] = time[-1]
        elif name in rev_attrs_map and rev_attrs_map[name] in ds_attrs:
            attrs[name] = ds_attrs[rev_attrs_map[name]].strip()
        elif name in ds_attrs:
            attrs[name] = ds_attrs[name].strip()

    logger.debug(f"Got fields {attrs.keys()} from file.")
    return attrs


def _parse_from_zarr(path: os.PathLike, get_vars=True):
    """Obtain the list of variables and the list of global attributes from a zarr dataset, reading the JSON files directly.

    Variables are those
    - where .zattrs/_ARRAY_DIMENSIONS is not empty
    - where .zattrs/_ARRAY_DIMENSIONS does not contain the variable name
    - who do not appear in any "coordinates" attribute.
    """
    path = Path(path)

    if (path / ".zattrs").is_file():
        with (path / ".zattrs").open() as f:
            ds_attrs = json.load(f)
    else:
        ds_attrs = {}

    variables = []
    if get_vars:
        coords = []
        for varpath in path.iterdir():
            if varpath.is_dir() and (varpath / ".zattrs").is_file():
                with (varpath / ".zattrs").open() as f:
                    var_attrs = json.load(f)
                if (
                    varpath.name in var_attrs["_ARRAY_DIMENSIONS"]
                    or len(var_attrs["_ARRAY_DIMENSIONS"]) == 0
                ):
                    coords.append(varpath.name)
                if "coordinates" in var_attrs:
                    coords.extend(
                        list(map(str.strip, var_attrs["coordinates"].split(" ")))
                    )
        variables = [
            varpath.name
            for varpath in path.iterdir()
            if varpath.name not in coords and varpath.is_dir()
        ]
    return ds_attrs, variables


def _parse_from_nc(path: os.PathLike, get_vars=True, get_time=True):
    """Obtain the list of variables, the time coordinate, and the list of global attributes from a netCDF dataset, using netCDF4."""
    ds = netCDF4.Dataset(str(path))
    ds_attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}

    variables = []
    if get_vars:
        coords = []
        for name, var in ds.variables.items():
            if "coordinates" in var.ncattrs():
                coords.extend(
                    list(map(str.strip, var.getncattr("coordinates").split(" ")))
                )
            if len(var.dimensions) == 0 or name in var.dimensions:
                coords.append(name)
        variables = [var for var in ds.variables.keys() if var not in coords]

    time = None
    if get_time and "time" in ds.variables:
        time = xr.CFTimeIndex(
            cftime.num2date(
                ds["time"][:], calendar=ds["time"].calendar, units=ds["time"].units
            ).data
        )
    ds.close()
    return ds_attrs, variables, time


# ## Path building ## #

# vv Functions below are almost identical to miranda vv
# Differences:
#  - The signature of `date_parser` is slightly different
#  - We have access to `xrfreq` instead of `frequency`.


def _parse_option(option: dict, facets: dict):
    """Parse an option element of the facet schema tree."""
    facet_value = facets[option["option"]]
    if "value" in option:
        if isinstance(option["value"], str):
            answer = facet_value == option["value"]
        else:  # A list
            answer = facet_value in option["value"]
    else:
        answer = not isna(facet_value)

    if "is_true" in option and answer:
        return option["is_true"]
    if "else" in option and not answer:
        return option["else"]
    return answer


def _parse_level(schema: Union[dict, str], facets: dict):
    if isinstance(schema, str):
        if schema == "DATES":
            return _parse_dates(facets)

        # A single facet:
        if isna(facets[schema]):
            return None
        return facets[schema]
    if isinstance(schema, list):
        parts = []
        for element in schema:
            part = _parse_level(element, facets)
            if not isna(part):
                parts.append(part)
        return "_".join(parts)
    if "option" in schema:
        answer = _parse_option(schema, facets)
        if isinstance(answer, bool) and not answer:
            # Test failed with no "else" value, we skip this level.
            return None
        return _parse_level(answer, facets)
    if "text" in schema:
        return schema["text"]
    raise ValueError(f"Invalid schema : {schema}")


def _parse_dates(facets):
    if facets["xrfreq"] == "fx":
        return "fx"

    start = date_parser(facets["date_start"], out_dtype="datetime")
    end = date_parser(facets["date_end"], out_dtype="datetime")
    freq = pd.Timedelta(CV.xrfreq_to_timedelta(facets["xrfreq"]))

    # Full years : Starts on Jan 1st and is either annual or ends on Dec 31st (accepting Dec 30 for 360 cals)
    if (
        start.month == 1
        and start.day == 1
        and (
            freq >= pd.Timedelta(CV.xrfreq_to_timedelta("YS"))
            or (end.month == 12 and end.day > 29)
        )
    ):
        if start.year == end.year:
            return f"{start:%Y}"
        return f"{start:%Y}-{end:%Y}"
    # Full months : Starts on the 1st and is either montly or ends on the last day
    if start.day == 1 and (
        freq >= pd.Timedelta(CV.xrfreq_to_timedelta("M")) or end.day > 27
    ):
        # Full months
        if (start.year, start.month) == (end.year, end.month):
            return f"{start:%Y%m}"
        return f"{start:%Y%m}-{end:%Y%m}"
    # The full range
    return f"{start:%Y%m%d}-{end:%Y%m%d}"


def _parse_filename(schema: list, facets: dict) -> str:
    return "_".join(
        [
            facets[element] if element != "DATES" else _parse_dates(facets)
            for element in schema
            if element == "DATES" or not isna(facets[element])
        ]
    )


def _parse_structure(schema: list, facets: dict) -> list:
    folder_tree = list()
    for level in schema:
        part = _parse_level(level, facets)
        if not isna(part):
            folder_tree.append(part)
    return folder_tree


# ^^ Functions above are almost identical to miranda ^^


def parse_schema(facets: dict, schema: list) -> tuple[list[str], str]:
    """Parse the schema from a configuration and construct path using a dictionary of facets.

    Parameters
    ----------
    facets : dict
    schema : List of structures.

    Returns
    -------
    list of folders, filename without suffix
    """
    for i, structure in enumerate(schema):
        if {"with", "structure", "filename"} != set(structure.keys()):
            raise ValueError("Invalid schema specification.")

        match = reduce(
            op.and_, map(partial(_parse_option, facets=facets), structure["with"])
        )
        if match:
            return (
                _parse_structure(structure["structure"], facets),
                _parse_filename(structure["filename"], facets),
            )
    raise ValueError(f"This file doesn't match any structure. Facets:\n{facets}")


@parse_config
def restructure_files(
    catalog: Union[DataCatalog, pd.DataFrame],
    folder: Union[str, os.PathLike] = ".",
    schema: Optional[Union[str, os.PathLike, dict]] = None,
    category: str = "raw",
    copy: bool = True,
    overwrite: bool = False,
) -> Union[DataCatalog, pd.DataFrame]:
    """Restructure datasets based on a schema.

    If the "variable" field appears in the path, multi-variable datasets are split into single
    variable datasets.

    Parameters
    ----------
    catalog: DataCatalog
        Catalog or DataFrame of files to restructure.
    folder : str or os.PathLike
        Parent folder on which to extend the filetree structure.
    schema : str or os.PathLike, optional
        Path to YAML schematic of database structure. If None, will use a basic schema.
        See the comments in the `xscen/data/base_schema.yml` file for more details on its construction.
    category : {raw, derived}
        Category of the path. Not to be confused with "processing_level" which is more granular.
    copy: bool
        Whether to copy (True, default) or move (False) files into the new location.
    overwrite: bool
        If True, overwrite existing files. If False (default), an error is raised instead.

    Notes
    -----
    If `copy` is True and the function fails during the copy, the destination is erased
    before reraising the exception. The folder hierarchy will not be erased, only the final dataset (zarr or nc).
    If `copy` is False, both source and destination could end up corrupted on an I/O failure.

    Returns
    -------
    same as `catalog`
        A copy of the initial catalog with the new paths. Index will be inconsistent if multi-variables dataset have been splitted.
    """
    if isinstance(catalog, esm_datastore):
        df = catalog.df
    else:
        df = catalog

    if not isinstance(schema, dict):
        if Path(schema).is_file():
            with open(schema).open() as f:
                schema = yaml.safe_load(f)
        else:
            try:
                res = urlopen(schema)
            except HTTPError:
                raise ValueError(
                    f"Passed schema ({schema}) is neither an existing path or a valid url."
                )
            else:
                schema = yaml.safe_load(res)

    structures = schema[category]

    for i, row in df.iterrows():
        path_in = row.path
        if isinstance(row.variable, (tuple, list)):
            paths = {}
            for variable in row.variable:
                facets = dict(**row) | {"variable": variable}
                p, f = parse_schema(facets, structures)
                paths[variable] = Path(folder).joinpath("/".join(p)) / f
            if len(set(paths.values())) == 1:
                path_out = paths[variable]
            else:
                path_out = paths
        else:
            p, f = parse_schema(row, structures)
            path_out = Path(folder).joinpath("/".join(p)) / f

        copy_dataset(path_in, path_out)
