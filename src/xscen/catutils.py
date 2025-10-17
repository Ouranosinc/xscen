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
from itertools import chain, combinations, product
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cftime
import netCDF4
import numpy as np
import pandas as pd
import parse
import xarray as xr
import yaml
import zarr
from intake_esm import esm_datastore
from pandas import isna

from .catalog import COLUMNS, DataCatalog, generate_id
from .config import parse_config
from .io import get_engine
from .utils import CV, date_parser, ensure_new_xrfreq, get_cat_attrs


logger = logging.getLogger(__name__)


__all__ = [
    "build_path",
    "parse_directory",
    "parse_from_ds",
    "patterns_from_schema",
    "register_parse_type",
]
# ## File finding and path parsing ## #


SUFFIX_TO_FORMAT = {
    ".nc": "nc",
    ".nc4": "nc",
    ".zip": "zarr",
    ".zarr.zip": "zarr",
    ".zarr": "zarr",
}
"""Mapping from file suffix to format.

This is used to populate the "format" esm catalog column from the parsed path.
"""

EXTRA_PARSE_TYPES = {}
"""Extra parse types to add to parse's default.

Add your own types with the :py:func:`register_parse_type` decorator.
"""


def register_parse_type(name: str, regex: str = r"([^\_\/\\]*)", group_count: int = 1):
    r"""
    Register a new parse type to be available in :py:func:`parse_directory` patterns.

    Function decorated by this will be registered in :py:data:`EXTRA_PARSE_TYPES`.
    The function must take a single string and should return a single string.
    If you return a different type, it may interfere with the other steps of `parse_directory`.

    Parameters
    ----------
    name: str
        The type name. To make use of this type, put "{field:name}" in your pattern.
    regex: str
        A regex string to determine what can be matched by this type.
        The default matches anything but / \ and _, same as the default parse type.
    group_count: int
        The number of regex groups in the previous regex string.
    """

    def _register_parse_type(func):
        EXTRA_PARSE_TYPES[name] = parse.with_pattern(regex, regex_group_count=group_count)(func)
        return func

    return _register_parse_type


@register_parse_type("no_", regex=r"([^\_\/\\]*)", group_count=1)
def _parse_word(text: str) -> str:
    r"""Parse helper to match strings with anything except / \ or _."""
    return text


@register_parse_type("_", regex=r"([^\/\\]*)", group_count=1)
def _parse_level(text: str) -> str:
    r"""Parse helper to match strings with anything except / or \."""
    return text


# Minimum 4 digits for a date (a single year). Maximum is, in theory, YYYYMMDDHHMMSS so 14.
@register_parse_type("datebounds", regex=r"(([\d]{4,15}(\-[\d]{4,15})?)|fx)", group_count=3)
def _parse_datebounds(
    text: str,
) -> list[str] | tuple[None, None] | tuple[str, str]:
    """Parse helper to translate date bounds, used in the special DATES field."""
    if "-" in text:
        return text.split("-")
    if text == "fx":
        return None, None
    return text, text


def _find_assets(
    root: str | os.PathLike,
    exts: set[str],
    lengths: set[int],
    dirglob: str | None = None,
    skip_dirs: list[os.PathLike] | None = None,
):
    """
    Walk recursively over files in a directory, filtering according to a glob pattern, path depth and extensions.

    Parameters
    ----------
    root : str or Pathlike
        Path of the directory to walk through.
    exts : set of strings
        Set of file extensions to look for.
    lengths : set of ints
        Set of path depths to look for.
    dirglob : str, optional
        A glob pattern. If given, only parent folders matching this pattern are walked through.
        This pattern can not include the asset's basename.
    skip_dirs : list of Paths, optional
        A list of directories to skip on the walk.
    """
    skip_dirs = skip_dirs or []
    root = str(Path(root))  # to be sure
    for top, alldirs, files in os.walk(root):
        # Split zarr subdirectories from next iteration
        zarrs = []
        for dr in deepcopy(alldirs):
            fdr = Path(top).joinpath(dr)
            if dr.endswith(".zarr"):
                zarrs.append(dr)
                alldirs.remove(dr)
            if fdr in skip_dirs:
                logger.debug("Skipping %s", fdr)
                alldirs.remove(dr)

        if top != root and (os.path.relpath(top, root).count(os.path.sep) + 1) not in lengths:
            continue

        if dirglob is not None and not fnmatch(top, dirglob):
            continue

        if ".zarr" in exts:
            for zr in zarrs:
                yield Path(top).joinpath(zr).as_posix()
        if exts - {".zarr"}:  # There are more exts than
            for file in files:
                if Path(file).suffix in exts:
                    yield Path(top).joinpath(file).as_posix()


def _compile_pattern(pattern: str) -> parse.Parser:
    r"""
    Compile a parse pattern (if needed) for quicker evaluation.

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
    path: os.PathLike | str,
    root: os.PathLike | str,
    patterns: list[str | parse.Parser],
    read_from_file: list[str] | dict | None = None,
    attrs_map: dict | None = None,
    xr_open_kwargs: dict | None = None,
) -> dict | None:
    """
    Extract metadata information from the file path.

    Parameters
    ----------
    path : os.PathLike or str
        Full file path.
    root : os.PathLike or str
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
    path = abs_path.relative_to(Path(root))
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
    d["format"] = SUFFIX_TO_FORMAT.get(path.suffix, path.suffix[1:])

    if "DATES" in d:
        d["date_start"], d["date_end"] = d.pop("DATES")

    if read_from_file:
        fromfile = parse_from_ds(abs_path, names=read_from_file, attrs_map=attrs_map, **xr_open_kwargs)
        d.update(fromfile)

    # files with a single year/month
    if "date_end" not in d and "date_start" in d:
        d["date_end"] = d["date_start"]

    # strip to clean off lost spaces and line jumps
    # do not include wildcarded fields (? was transformed to _ in _compile_pattern)
    return {k: v.strip() if isinstance(v, str) else v for k, v in d.items() if not k.startswith("_")}


def _parse_dir(  # noqa: C901
    root: os.PathLike | str,
    patterns: list[str],
    dirglob: str | None = None,
    skip_dirs: list[os.PathLike] | None = None,
    checks: list[str] | None = None,
    read_from_file: list[str] | dict | None = None,
    attrs_map: dict | None = None,
    xr_open_kwargs: dict | None = None,
    progress: bool = False,
):
    """
    Iterate and parses files in a directory, filtering according to basic pattern properties and optional checks.

    Parameters
    ----------
    root : os.PathLike or str
        Path to walk through.
    patterns : list of strings or compiled parsers
        Patterns that the files will be checked against.
        The extensions of the patterns are extracted and only paths with these are returned.
        Also, the depths of the patterns are calculated and only paths of this depth under the root are returned.
    dirglob : str
        A glob pattern. If given, only parent folders matching this pattern are walked through.
        This pattern can not include the asset's basename.
    skip_dirs : list of strings or Paths, optional
        A list of directories to skip in the walk.
    checks: list of str, optional
        A list of checks to perform, available values are:
        - "readable" : Check that the file is readable by the current user.
        - "writable" : Check that the file is writable by the current user.
        - "ncvalid" : For netCDF, check that it is valid (openable with netCDF4).
        All checks will slow down the parsing.
    read_from_file : list of string or dict, optional
        If not None, passed directly to :py:func:`parse_from_ds` as `names`.
        If None (default), only the path is parsed, the file is not opened.
    attrs_map : dict, optional
        If `read_from_file` is not None, passed directly to :py:func:`parse_from_ds`.
    xr_open_kwargs : dict, optional
        If `read_from_file` is not None, passed directly to :py:func:`parse_from_ds`.
    progress : bool
        If True, the number of found files is printed to stdout.

    Returns
    -------
    List of dictionaries
        Metadata parsed from each found asset.
    """
    lengths = {patt.count(os.path.sep) for patt in patterns}
    exts = {Path(patt).suffix for patt in patterns}
    comp_patterns = list(map(_compile_pattern, patterns))
    checks = checks or []
    parsed = []

    root = Path(root)
    if any([(skd in root.parents) or (skd == root) for skd in (skip_dirs or [])]):
        logger.debug("Skipping %s", root)
        return parsed

    # Multithread, communicating via FIFO queues.
    # This thread walks the directory
    # Another thread runs the checks
    # Another thread parses the path and file.
    # In theory, for a local disk, walking a directory cannot be parallelized. This is not as true for network-mounted drives.
    # Thus, we parallelize the parsing steps.
    # If the name-parsing step becomes blocking, we could try to increase the number of threads (but netCDF4 can't multithread...)
    # Usually, the walking is the bottleneck.
    q_found = queue.Queue()
    q_checked = queue.Queue()

    def check_worker():
        # Worker that processes the checks.
        while True:
            path = q_found.get()
            valid = True
            if "readable" in checks and not os.access(path, os.R_OK):
                valid = False
            if "writable" in checks and not os.access(path, os.W_OK):
                valid = False
            if "ncvalid" in checks:
                try:  # Simple check that the file is openable
                    if get_engine(path) == "netcdf4":
                        # if get_engine is "h5netcdf", it means h5py was able to recognize it.
                        # TODO: testing for zarr validity is not implemented
                        with netCDF4.Dataset(path):
                            pass
                # FIXME: This is a catch-all, we should catch the specific exception raised by netCDF4.
                except Exception:  # noqa: BLE001
                    valid = False
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
            # FIXME: This is not specific enough, we should catch the specific exception raised by _name_parser.
            except Exception as err:  # noqa: BLE001
                msg = f"Parsing file {path} failed with {err}."
                logger.error(msg)
            else:
                if d is not None:
                    parsed.append(d)
                    n = len(parsed)
                    # Print number of files but on round numbers to limit the calls to stdout for large collections
                    if progress and all([(n < N or (n % N == 0)) for N in [10, 100, 1000]]):
                        print(f"Found {n:7d} files", end="\r")
                else:
                    msg = f"File {path} didn't match any pattern."
                    logger.debug(msg)
            q_checked.task_done()

    CW = threading.Thread(target=check_worker, daemon=True)
    CW.start()

    PW = threading.Thread(target=parse_worker, daemon=True)
    PW.start()

    # Skip the checks if none are requested (save some overhead)
    q = q_found if checks else q_checked
    for path in _find_assets(Path(root), exts, lengths, dirglob, skip_dirs):
        q.put(path)

    q_found.join()
    q_checked.join()
    return parsed


def _get_new_item(name, newval, repval, oldval, fromcol, is_list):
    if is_list:
        if name == fromcol:  # We replace only the repval element of the list
            return tuple(newval if v == repval else v for v in oldval)
        # We must return a tuple, replace the whole list with a single element.
        return (newval,)
    return newval  # Simple replacement


def _replace_in_row(oldrow: pd.Series, replacements: dict):
    """
    Replace values in Series (row) according to replacements mapping.

    Replacements can be simple mappings, but also mapping to other fields.
    List-like fields are handled.
    """
    row = oldrow.copy()
    list_cols = [col for col in oldrow.index if isinstance(oldrow[col], tuple | list)]
    for col, reps in replacements.items():
        if col not in row:
            continue
        for repval, new in reps.items():
            # Either the field is a list containing the value to replace, or it is the value to replace.
            if (col in list_cols and repval in row[col]) or repval == row[col]:
                if isinstance(new, dict):  # Replacement is for multiple columns
                    for name, newval in new.items():
                        row[name] = _get_new_item(name, newval, repval, row[col], col, name in list_cols)
                else:
                    row[col] = _get_new_item(col, new, repval, row[col], col, col in list_cols)
    # Special case for "variable" where we remove Nones.
    if "variable" in row and "variable" in list_cols and None in row["variable"]:
        row["variable"] = tuple(v for v in row["variable"] if v is not None)
    return row


def _parse_first_ds(grp: pd.DataFrame, cols: list[str], attrs_map: dict, xr_open_kwargs: dict):
    """Parse attributes from one file per group, apply them to the whole group."""
    fromfile = parse_from_ds(grp.path.iloc[0], cols, attrs_map, **xr_open_kwargs)

    msg = f"Got {len(fromfile)} fields, applying to {len(grp)} entries."
    logger.info(msg)
    out = grp.copy()
    for col, val in fromfile.items():
        for i in grp.index:  # If val is an iterable we can't use loc.
            out.at[i, col] = val
    return out


@parse_config
def parse_directory(  # noqa: C901
    directories: str | list[str | os.PathLike],
    patterns: list[str],
    *,
    id_columns: list[str] | None = None,
    read_from_file: (bool | Sequence[str] | tuple[Sequence[str], Sequence[str]] | Sequence[tuple[Sequence[str], Sequence[str]]]) = False,
    homogenous_info: dict | None = None,
    cvs: str | os.PathLike | dict | None = None,
    dirglob: str | None = None,
    skip_dirs: list[str | os.PathLike] | None = None,
    xr_open_kwargs: Mapping[str, Any] | None = None,
    only_official_columns: bool = True,
    progress: bool = False,
    parallel_dirs: bool | int = False,
    file_checks: list[str] | None = None,
) -> pd.DataFrame:
    r"""
    Parse files in a directory and return them as a pd.DataFrame.

    Parameters
    ----------
    directories : list of os.PathLike or list of str
        List of directories to parse. The parse is recursive.
    patterns : list of str
        List of possible patterns to be used by :py:func:`parse.parse` to decode the file names. See Notes below.
    id_columns : list of str, optional
        List of column names on which to base the dataset definition. Empty columns will be skipped.
        If None (default), it uses :py:data:`ID_COLUMNS`.
    read_from_file : boolean or set of strings or tuple of 2 sets of strings or list of tuples
        If True, if some fields were not parsed from their path, files are opened and
        missing fields are parsed from their metadata, if found.
        If a sequence of column names, only those fields are parsed from the file, if missing.
        If False (default), files are never opened.
        If a tuple of 2 lists of strings, only the first file of groups defined by the first list of
        columns is read and the second list of columns is parsed from the file and applied to the whole group.
        For example, `(["source"],["institution", "activity"])` will find a group with all the files
        that have the same source, open only one of the files to read the institution
        and activity, and write this information in the catalog for all files of the group.
        It can also be a list of those tuples.
    homogenous_info : dict, optional
        Using the {column_name: description} format, information to apply to all files.
        These are applied before the `cvs`.
    cvs: str or os.PathLike or dict, optional
        Dictionary with mapping from parsed term to preferred terms (Controlled VocabularieS) for each column.
        May have an additional "attributes" entry which maps from attribute names in the files to
        official column names. The attribute translation is done before the rest.
        In the "variable" entry, if a name is mapped to None (null), that variable will not be listed in the catalog.
        A term can map to another mapping from field name to values, so that a value on one column triggers the filling of other columns.
        In the latter case, that other column must exist beforehand, whether it was in the pattern or in the homogenous_info.
    dirglob : str, optional
        A glob pattern for path matching to accelerate the parsing of a directory tree if only a subtree is needed.
        Only folders matching the pattern are parsed to find datasets.
    skip_dirs : list of str or Paths, optional
        A list of folders that will be removed from the search, should be absolute.
    xr_open_kwargs: dict
        If needed, arguments to send xr.open_dataset() when opening the file to read the attributes.
    only_official_columns: bool
        If True (default), this ensures the final catalog only has the columns defined in :py:data:`xscen.catalog.COLUMNS`.
        Other fields in the patterns will raise an error.
        If False, the columns are those used in the patterns and the homogeneous info.
        In that case, the column order is not determined.
        Path, format, and id are always present in the output.
    progress : bool
        If True, a counter is shown in stdout when finding files on disk. Does nothing if `parallel_dirs` is not False.
    parallel_dirs: bool or int
        If True, each directory is searched in parallel. If an int, it is the number of parallel searches.
        This should only be significantly useful if the directories are on different disks.
    file_checks: list of str, optional
        A list of file checks to run on the parsed files. Available values are:
        - "readable" : Check that the file is readable by the current user.
        - "writable" : Check that the file is writable by the current user.
        - "ncvalid" : For netCDF, check that it is valid (openable with netCDF4).
        Any check will slow down the parsing.

    Notes
    -----
    - Official columns names are controlled and ordered by :py:data:`COLUMNS`:
        ["id", "type", "processing_level", "mip_era", "activity", "driving_model", "driving_member", "institution",
         "source", "bias_adjust_institution", "bias_adjust_project", "bias_adjust_reference", "experiment", "member",
         "xrfreq", "frequency", "variable", "domain", "date_start", "date_end", "version"]
    - Not all column names have to be present, but "xrfreq" (obtainable through "frequency"), "variable",
        "date_start" and "processing_level" are necessary for a workable catalog.
    - 'patterns' should highlight the columns with braces.
        This acts like the reverse operation of `format()`. It is a template string with `{field name:type}` elements.
        The default "type" will match alphanumeric parts of the path, excluding the "_", "/" and "\" characters.
        The "_" type will allow underscores.
        Field names prefixed by "?" will not be included in the output.
        See the documentation of :py:mod:`parse` for more type options.
        You can also add your own types using the :py:func:`register_parse_type` decorator.

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
    if isinstance(directories, str | Path):
        directories = [directories]
    homogenous_info = homogenous_info or {}
    xr_open_kwargs = xr_open_kwargs or {}
    if only_official_columns:
        columns = set(COLUMNS) - homogenous_info.keys()
        pattern_fields = {f for f in set.union(*(set(patt.named_fields) for patt in map(_compile_pattern, patterns))) if not f.startswith("_")} - {
            "DATES"
        }
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
            with Path(cvs).open(encoding="utf-8") as f:
                cvs = yaml.safe_load(f)
        attrs_map = cvs.pop("attributes", {})
    else:
        attrs_map = {}

    parse_kwargs = dict(
        patterns=patterns,
        dirglob=dirglob,
        skip_dirs=[Path(d) for d in (skip_dirs or [])],
        read_from_file=read_from_file if not read_file_groups else None,
        attrs_map=attrs_map,
        xr_open_kwargs=xr_open_kwargs,
        checks=file_checks,
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
            parsed.extend(_parse_dir(directory, progress=progress, **parse_kwargs))

    if not parsed:
        raise ValueError("No files found.")
    else:
        if progress:
            print()  # This is because of the \r outputted in the _parse_dir call.
        msg = f"Found and parsed {len(parsed)} files."
        logger.info(msg)

    # Path has become NaN when some paths didn't fit any passed pattern
    df = pd.DataFrame(parsed).dropna(axis=0, subset=["path"])

    if only_official_columns:  # Add the missing official columns
        for col in set(COLUMNS) - set(df.columns):
            df[col] = None

    if read_file_groups:  # Read fields from file, but only one per group.
        for group_cols, parse_cols in read_from_file:
            df = (  # column indexing avoids a deprecation where cols in group_cols are not sent to apply
                df.groupby(group_cols)[df.columns]
                .apply(
                    _parse_first_ds,
                    cols=parse_cols,
                    attrs_map=attrs_map,
                    xr_open_kwargs=xr_open_kwargs,
                )
                .reset_index(drop=True)
            )

    # Everything below could be wrapped in a function to be applied to each row maybe allowing some basic parallelization with dask (or else).
    # Add homogeous info
    for key, val in homogenous_info.items():
        df[key] = val

    # Replace entries by definitions found in CV
    if cvs:
        df = df.apply(_replace_in_row, axis=1, replacements=cvs)

    # Fix potential legacy xrfreq
    if "xrfreq" in df.columns:
        df["xrfreq"] = df["xrfreq"].apply(ensure_new_xrfreq)

    # translate xrfreq into frequencies and vice-versa
    if {"xrfreq", "frequency"}.issubset(df.columns):
        df.fillna(
            {"xrfreq": df["frequency"].apply(CV.frequency_to_xrfreq, default=pd.NA)},
            inplace=True,
        )
        df.fillna(
            {"frequency": df["xrfreq"].apply(CV.xrfreq_to_frequency, default=pd.NA)},
            inplace=True,
        )

    # Parse dates
    # If we don't do the to_numpy(na_value=np.datetime64('')).astype('<M8[ms]') trick,
    # the dtype will be "object" if any of the dates are out-of-bounds.
    # `na_values=np.datetime64('')` is needed because pandas' NaT does not translate to numpy's NaT, but to float.
    if "date_start" in df.columns:
        df["date_start"] = df["date_start"].apply(date_parser).to_numpy(na_value=np.datetime64("")).astype("<M8[ms]")
    if "date_end" in df.columns:
        df["date_end"] = df["date_end"].apply(date_parser, end_of_period=True).to_numpy(na_value=np.datetime64("")).astype("<M8[ms]")
    # Checks
    if {"date_start", "date_end", "xrfreq", "frequency"}.issubset(df.columns):
        # All NaN dates correspond to a fx frequency.
        invalid = df.date_start.isnull() & df.date_end.isnull() & (df.xrfreq != "fx")
        n = invalid.sum()
        if n > 0:
            warnings.warn(f"{n} invalid entries where the start and end dates are Null but the frequency is not 'fx'.", stacklevel=2)
            msg = f"Paths: {df.path[invalid].values}"
            logger.debug(msg)
            df = df[~invalid]
        # Exact opposite
        invalid = df.date_start.notnull() & df.date_end.notnull() & (df.xrfreq == "fx")
        n = invalid.sum()
        if n > 0:
            warnings.warn(f"{n} invalid entries where the start and end dates are given but the frequency is 'fx'.", stacklevel=2)
            msg = f"Paths: {df.path[invalid].values}"
            logger.debug(msg)
            df = df[~invalid]

    # Create id from user specifications
    df["id"] = generate_id(df, id_columns)

    # TODO: ensure variable is a tuple ?

    # ensure path is a string
    df["path"] = df.path.apply(str)

    # Sort columns and return
    if only_official_columns:
        return df.loc[:, COLUMNS]
    return df


def parse_from_ds(  # noqa: C901
    obj: str | os.PathLike | xr.Dataset,
    names: Sequence[str],
    attrs_map: Mapping[str, str] | None = None,
    **xrkwargs,
):
    """
    Parse a list of catalog fields from the file/dataset itself.

    If passed a path, this opens the file.

    Infers the variable from the variables.
    Infers xrfreq, frequency, date_start and date_end from the time coordinate if present.
    Infers other attributes from the coordinates or the global attributes. Attributes names
    can be translated using the `attrs_map` mapping (from file attribute name to name in `names`).

    If the obj is the path to a Zarr dataset and none of "frequency", "xrfreq", "date_start" or "date_end"
    are requested, :py:func:`parse_from_zarr` is used instead of opening the file.

    Parameters
    ----------
    obj: str or os.PathLike or xr.Dataset
        Dataset to parse.
    names: sequence of str
        List of attributes to be parsed from the dataset.
    attrs_map: dict, optional
        In the case of non-standard names in the file, this can be used to match entries in the files to specific 'names' in the requested list.
    xrkwargs:
        Arguments to be passed to open_dataset().
    """
    get_time = bool({"frequency", "xrfreq", "date_start", "date_end"}.intersection(names))
    if not isinstance(obj, xr.Dataset):
        obj = Path(obj)

    if isinstance(obj, Path) and obj.suffixes[-1] == ".zarr":
        msg = f"Parsing attributes from Zarr {obj}."
        logger.info(msg)
        ds_attrs, variables, time = _parse_from_zarr(obj, get_vars="variable" in names, get_time=get_time)
    elif isinstance(obj, Path) and obj.suffixes[-1] == ".nc":
        msg = f"Parsing attributes with netCDF4 from {obj}."
        logger.info(msg)
        ds_attrs, variables, time = _parse_from_nc(obj, get_vars="variable" in names, get_time=get_time)
    else:
        if isinstance(obj, Path):
            msg = f"Parsing attributes with xarray from {obj}."
            logger.info(msg)
            obj = xr.open_dataset(obj, engine=get_engine(obj), **xrkwargs)
        ds_attrs = obj.attrs
        time = obj.indexes["time"] if "time" in obj else None
        variables = set(obj.data_vars.keys()).difference([v for v in obj.data_vars if len(obj[v].dims) == 0])

    rev_attrs_map = {v: k for k, v in (attrs_map or {}).items()}
    attrs = {}

    for name in names:
        if name == "variable":
            attrs["variable"] = tuple(sorted(variables))
        elif name in ("frequency", "xrfreq") and time is not None and time.size > 3:
            # round to the minute to catch floating point imprecision
            freq = xr.infer_freq(time.round("min"))
            if freq:
                if "xrfreq" in names:
                    attrs["xrfreq"] = freq
                if "frequency" in names:
                    attrs["frequency"] = CV.xrfreq_to_frequency(freq)
            else:
                warnings.warn(f"Couldn't infer frequency of dataset {obj if not isinstance(obj, xr.Dataset) else ''}", stacklevel=2)
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

    msg = f"Got fields {attrs.keys()} from file."
    logger.debug(msg)
    return attrs


def _parse_from_zarr(path: os.PathLike | str, get_vars: bool = True, get_time: bool = True):
    """
    Obtain the list of variables, the time coordinate and the list of global attributes from a zarr dataset.

    Vars and attrs from reading the JSON files directly, time by reading the data with zarr.

    Variables are those
    - where .zattrs/_ARRAY_DIMENSIONS is not empty
    - where .zattrs/_ARRAY_DIMENSIONS does not contain the variable name
    - who do not appear in any "coordinates" attribute.

    Parameters
    ----------
    path: os.PathLike or str
        Path to the zarr dataset.
    get_vars: bool
        If True, return the list of variables.
    get_time: bool
        If True, return the time coordinate.
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
                if varpath.name in var_attrs["_ARRAY_DIMENSIONS"] or len(var_attrs["_ARRAY_DIMENSIONS"]) == 0:
                    coords.append(varpath.name)
                if "coordinates" in var_attrs:
                    coords.extend(list(map(str.strip, var_attrs["coordinates"].split(" "))))
        variables = [varpath.name for varpath in path.iterdir() if varpath.name not in coords and varpath.is_dir()]
    time = None
    if get_time and (path / "time").is_dir():
        ds = zarr.open(path)
        time = xr.CFTimeIndex(
            cftime.num2date(
                ds.time[:],
                calendar=ds.time.attrs["calendar"],
                units=ds.time.attrs["units"],
            )
        )
    return ds_attrs, variables, time


def _parse_from_nc(path: os.PathLike | str, get_vars: bool = True, get_time: bool = True):
    """
    Obtain the list of variables, the time coordinate, and the list of global attributes from a netCDF dataset, using netCDF4.

    Parameters
    ----------
    path: os.PathLike or str
        Path to the netCDF dataset.
    get_vars: bool
        If True, return the list of variables.
    get_time: bool
        If True, return the time coordinate.
    """
    ds = netCDF4.Dataset(str(Path(path)))
    ds_attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}

    variables = []
    if get_vars:
        coords = []
        for name, var in ds.variables.items():
            if "coordinates" in var.ncattrs():
                coords.extend(list(map(str.strip, var.getncattr("coordinates").split(" "))))
            if len(var.dimensions) == 0 or name in var.dimensions:
                coords.append(name)
        variables = [var for var in ds.variables.keys() if var not in coords]

    time = None
    if get_time and "time" in ds.variables:
        time = xr.CFTimeIndex(cftime.num2date(ds["time"][:], calendar=ds["time"].calendar, units=ds["time"].units).data)
    ds.close()
    return ds_attrs, variables, time


# ## Path building ## #
def _schema_option(option: dict, facets: dict):
    """Parse an option element of the facet schema tree."""
    facet_value = facets.get(option["facet"])
    if "value" in option:
        if isinstance(option["value"], str):
            answer = facet_value == option["value"]
        else:  # A list
            answer = facet_value in option["value"]
    else:
        answer = not isna(facet_value)
    return answer


def _schema_level(schema: dict | list[str] | str, facets: dict):
    if isinstance(schema, str):
        if schema.startswith("(") and schema.endswith(")"):
            optional = True
            schema = schema[1:-1]
        else:
            optional = False
        if schema == "DATES":
            return _schema_dates(facets, optional=optional)

        # A single facet:
        if isna(facets.get(schema)):
            if optional:
                return None
            raise ValueError(f"Facet {schema} is needed but None-like or missing in the data.")
        return facets[schema]
    if isinstance(schema, list):
        parts = []
        for element in schema:
            part = _schema_level(element, facets)
            if not isna(part):
                parts.append(part)
        return "_".join(parts)
    if isinstance(schema, dict) and "text" in schema:
        return schema["text"]
    raise ValueError(f"Invalid schema : {schema}")


def _schema_dates(facets: dict, optional: bool = False):
    if facets.get("xrfreq") == "fx":
        return "fx"

    if any([facets.get(f) is None for f in ["date_start", "date_end", "xrfreq"]]):
        if optional:
            return None
        raise ValueError("Facets date_start, date_end and xrfreq are needed, but at least one is missing or None-like in the data.")

    start = date_parser(facets["date_start"])
    end = date_parser(facets["date_end"])
    freq = pd.Timedelta(CV.xrfreq_to_timedelta(facets["xrfreq"]))

    # Full years : Starts on Jan 1st and is either annual or ends on Dec 31st (accepting Dec 30 for 360 cals)
    if start.month == 1 and start.day == 1 and (freq >= pd.Timedelta(CV.xrfreq_to_timedelta("YS")) or (end.month == 12 and end.day > 29)):
        if start.year == end.year:
            return f"{start:%4Y}"
        return f"{start:%4Y}-{end:%4Y}"
    # Full months : Starts on the 1st and is either monthly or ends on the last day
    if start.day == 1 and (freq >= pd.Timedelta(CV.xrfreq_to_timedelta("M")) or end.day > 27):
        # Full months
        if (start.year, start.month) == (end.year, end.month):
            return f"{start:%4Y%m}"
        return f"{start:%4Y%m}-{end:%4Y%m}"
    # The full range
    return f"{start:%4Y%m%d}-{end:%4Y%m%d}"


def _schema_filename(schema: list, facets: dict) -> str:
    return "_".join(
        [
            facets.get(element) if element != "DATES" else _schema_dates(facets)
            for element in schema
            if element == "DATES" or not isna(facets.get(element))
        ]
    )


def _schema_folders(schema: list, facets: dict) -> list:
    folder_tree = list()
    for level in schema:
        part = _schema_level(level, facets)
        if not isna(part):
            folder_tree.append(part)
    return folder_tree


def _get_needed_fields(schema: dict):
    """Return the list of facets that is needed for a given schema."""
    needed = set()
    for level in schema["folders"]:
        if isinstance(level, str):
            if not (level.startswith("(") and level.endswith(")")):
                needed.add(level)
        elif isinstance(level, list):
            for lvl in level:
                if not (lvl.startswith("(") and lvl.endswith(")")):
                    needed.add(lvl)
        elif not (isinstance(level, dict) and list(level.keys()) == ["text"]):
            raise ValueError(f"Invalid schema with unknown {level} of type {type(level)}.")
    return needed


def _read_schemas(schemas):
    if isinstance(schemas, dict) and {"folders", "filename"}.issubset(schemas.keys()):
        # Single schema
        # Remove any conditions (or insert empty one)
        schemas["with"] = []
        schemas = {"unnamed_schema": schemas}
    elif not isinstance(schemas, dict):
        if schemas is None:
            schemas = Path(__file__).parent / "data" / "file_schema.yml"
        with Path(schemas).open(encoding="utf-8") as f:
            schemas = yaml.safe_load(f)
    for name, schema in schemas.items():
        missing_fields = {"with", "folders", "filename"} - set(schema.keys())
        if missing_fields:
            raise ValueError(f"Invalid schema specification. Missing fields {missing_fields} in schema {name}.")
    return schemas


def _build_path(
    data: dict | xr.Dataset | xr.DataArray | pd.Series,
    schemas: dict,
    root: str | os.PathLike,
    get_type: bool = False,
    **extra_facets,
) -> Path | tuple[Path, str]:
    # Get all known metadata
    if isinstance(data, xr.Dataset | xr.DataArray):
        facets = (
            # Get non-attribute metadata
            parse_from_ds(data, ["frequency", "xrfreq", "date_start", "date_end", "variable"]) | data.attrs | get_cat_attrs(data)
        )
    elif isinstance(data, pd.Series):
        facets = dict(data)
    else:
        raise NotImplementedError(f"Cannot build path with object of type {type(data)}")

    facets = facets | extra_facets

    # Scalar-ize variable if needed.
    if "variable" in facets and not isinstance(facets["variable"], str) and len(facets["variable"]) == 1:
        facets["variable"] = facets["variable"][0]

    # Find the first fitting schema
    for name, schema in schemas.items():
        if not schema["with"]:
            match = True
        else:
            match = reduce(op.and_, map(partial(_schema_option, facets=facets), schema["with"]))
        if match:
            # Checks
            needed_fields = _get_needed_fields(schema)
            if missing_fields := needed_fields - set(facets.keys()):
                raise ValueError(f"Missing facets {missing_fields} are needed to build the path according to selected schema {name}.")
            if "variable" in needed_fields and not isinstance(facets["variable"], str):
                raise ValueError(
                    f"Selected schema {name} is meant to be used with single-variable datasets. Got multiple: {facets['variable']}. "
                    "You can override the facet by passing `variable='varname'` directly."
                )
            out = Path(*_schema_folders(schema["folders"], facets))
            out = out / _schema_filename(schema["filename"], facets)
            if root is not None:
                out = Path(root) / out
            if "format" in facets:  # Add extension
                # Can't use `with_suffix` in case there are dots in the name
                out = out.parent / f"{out.name}.{facets['format']}"
            if get_type:
                return out, name
            return out

    raise ValueError(f"This file doesn't match any schema. Facets:\n{facets}")


@parse_config
def build_path(
    data: dict | xr.Dataset | xr.DataArray | pd.Series | DataCatalog | pd.DataFrame,
    schemas: str | os.PathLike | dict | None = None,
    root: str | os.PathLike | None = None,
    **extra_facets,
) -> Path | DataCatalog | pd.DataFrame:
    r"""
    Parse the schema from a configuration and construct path using a dictionary of facets.

    Parameters
    ----------
    data : dict or xr.Dataset or xr.DataArray or pd.Series or DataCatalog or pd.DataFrame
        Dict of facets. Or xarray object to read the facets from. In the latter case, variable and time-dependent
        facets are read with :py:func:`parse_from_ds` and supplemented with all the object's attribute,
        giving priority to the "official" xscen attributes (prefixed with `cat:`, see :py:func:`xscen.utils.get_cat_attrs`).
        Can also be a catalog or a DataFrame, in which a "new_path" column is generated for each item.
    schemas : Path or dict, optional
        Path to YAML schematic of database schema. If None, will use a default schema.
        See the comments in the `xscen/data/file_schema.yml` file for more details on its construction.
        A dict of dict schemas can be given (same as reading the yaml).
        Or a single schema dict (single element of the yaml).
    root : str or Path, optional
        If given, the generated path(s) is given under this root one.
    \*\*extra_facets
        Extra facets to supplement or override metadadata missing from the first input.

    Returns
    -------
    Path or catalog
        Constructed path. If "format" is absent from the facets, it has no suffix.
        If `data` was a catalog, a copy with a "new_path" column is returned.
        Another "new_path_type" column is also added if `schemas` was a collection of schemas (like the default).

    Examples
    --------
    To rename a full catalog, the simplest way is to do:

    >>> import xscen as xs
    >>> import shutil as sh
    >>> new_cat = xs.catutils.build_path(old_cat)
    >>> for i, row in new_cat.iterrows():
    ...     sh.move(row.path, row.new_path)
    """
    if root:
        root = Path(root)
    schemas = _read_schemas(schemas)
    if isinstance(data, esm_datastore | pd.DataFrame):
        if isinstance(data, esm_datastore):
            df = data.df
        else:
            df = data

        df = df.copy()

        paths = df.apply(
            _build_path,
            axis=1,
            result_type="expand",
            schemas=schemas,
            root=root,
            get_type=True,
            **extra_facets,
        )
        df["new_path"] = paths[0].apply(str)
        if len(schemas) > 1:
            df["new_path_type"] = paths[1]
        return df
    return _build_path(data, schemas=schemas, root=root, get_type=False, **extra_facets)


def _as_template(a):
    return "{" + a + "}"


def partial_format(template, **fmtargs):
    """Format a template only partially, leaving un-formatted templates intact."""

    class PartialFormatDict(dict):
        def __missing__(self, key):
            return _as_template(key)

    return template.format_map(PartialFormatDict(**fmtargs))


def patterns_from_schema(schema: str | dict, exts: Sequence[str] | None = None):
    """
    Generate all valid patterns for a given schema.

    Generated patterns are meant for use with :py:func:`parse_directory`.
    This hardcodes the rule that facet can never contain a underscore ("_") except "variable".
    File names are not strict except for the date bounds element which must be at the end if present.

    Parameters
    ----------
    schema: dict or str
        A dict with keys "with" (optional), "folders" and "filename", constructed as described
        in the `xscen/data/file_schema.yml` file.
        Or the name of a pattern group from that file.
    exts: sequence of strings, optional
        A list of file extensions to consider, with the leading dot.
        Defaults to ``[".nc", ".zarr", ".zarr.zip"]``.

    Returns
    -------
    list of patterns compatible with :py:func:`parse_directory`.
    """
    if isinstance(schema, str):
        schemas = Path(__file__).parent / "data" / "file_schema.yml"
        with schemas.open(encoding="utf-8") as f:
            schema = yaml.safe_load(f)[schema]

    # # Base folder patterns

    # Index of optional folder parts
    opt_idx = [i for i, k in enumerate(schema["folders"]) if isinstance(k, str) and k.startswith("(")]

    raw_folders = []
    for skip in chain.from_iterable(combinations(opt_idx, r) for r in range(len(opt_idx) + 1)):
        # skip contains index of levels to skip
        # we go through every possible missing levels combinations
        parts = []
        for i, part in enumerate(schema["folders"]):
            if i in skip:
                continue
            if isinstance(part, str):
                if part.startswith("("):
                    part = part[1:-1]
                parts.append(_as_template(part))
            elif isinstance(part, dict):
                parts.append(part["text"])
            else:
                parts.append("_".join(map(_as_template, part)))
        raw_folders.append("/".join(parts))

    # # Inject conditions
    folders = raw_folders
    for conditions in schema["with"]:
        if "value" not in conditions:
            # This means that the facet must be set.
            # Not useful when parsing. Implicit with the facet in the pattern.
            continue

        # Ensure a list
        if isinstance(conditions["value"], str):
            value = [conditions["value"]]
        else:
            value = conditions["value"]

        patterns = []
        for patt in folders:
            for val in value:
                patterns.append(partial_format(patt, **{conditions["facet"]: val}))
        folders = patterns

    # # Inject parsing requirements (hardcoded :( )
    folders = [folder.replace("{variable}", "{variable:_}") for folder in folders]

    # # Filenames
    if "DATES" in schema["filename"]:
        if schema["filename"][-1] != "DATES":
            raise ValueError("Reverse pattern generation is not supported for filenames with date bounds not at the end.")
        filename = "{?:_}_{DATES}"
    else:
        filename = "{?:_}"

    exts = exts or [".nc", ".zarr", ".zarr.zip"]

    patterns = [f"{fold}/{filename}{ext}" for fold, ext in product(folders, exts)]

    return patterns
