import ast
import itertools
import json
import logging
import os
import re
import subprocess
import warnings
from copy import deepcopy
from datetime import datetime
from glob import glob
from pathlib import Path, PosixPath
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import cftime
import dask
import fsspec as fs
import intake_esm
import netCDF4
import pandas as pd
import xarray
import xarray as xr
import yaml
from dask.diagnostics import ProgressBar
from intake.source.utils import reverse_format
from intake_esm.cat import ESMCatalogModel

from . import CV
from .config import CONFIG, parse_config, recursive_update
from .io import get_engine

logger = logging.getLogger(__name__)
# Monkey patch for attribute names in the output of to_dataset_dict
intake_esm.set_options(attrs_prefix="cat")

"""Official column names."""
# As much as possible, these catalog columns and entries should align with: https://github.com/WCRP-CMIP/CMIP6_CVs and https://github.com/ES-DOC/pyessv-archive
# See docs/columns.rst for a description of each entry.
COLUMNS = [
    "id",
    "type",
    "processing_level",
    "bias_adjust_institution",
    "bias_adjust_project",
    "mip_era",
    "activity",
    "driving_institution",
    "driving_model",
    "institution",
    "source",
    "experiment",
    "member",
    "xrfreq",
    "frequency",
    "variable",
    "domain",
    "date_start",
    "date_end",
    "version",
    "format",
    "path",
]

"""Default columns used to create a unique ID"""
ID_COLUMNS = [
    "bias_adjust_project",
    "mip_era",
    "activity",
    "driving_model",
    "institution",
    "source",
    "experiment",
    "member",
    "domain",
]


"""Official ESM column data for the catalogs."""
_esm_col_data = {
    "esmcat_version": "0.1.0",  # intake-esm JSON file structure version, as per: https://github.com/NCAR/esm-collection-spec
    "assets": {"column_name": "path", "format_column_name": "format"},
    "aggregation_control": {
        "variable_column_name": "variable",
        "groupby_attrs": ["id", "domain", "processing_level", "xrfreq"],
        "aggregations": [
            {
                "type": "join_existing",
                "attribute_name": "date_start",
                "options": {"dim": "time"},
            },
            {"type": "union", "attribute_name": "variable"},
        ],
    },
    "attributes": [],
}


def parse_list_of_strings(elem):
    """Parse a element of a csv in case it is a list of strings."""
    if elem.startswith("(") or elem.startswith("["):
        out = ast.literal_eval(elem)
        return out
    return (elem,)


"""Offical kwargs to pass to `pd.read_csv` when opening an Ouranos catalog."""
csv_kwargs = {
    "dtype": {
        key: "category" if not key == "path" else "string[pyarrow]"
        for key in COLUMNS
        if key not in ["variable", "date_start", "date_end"]
    },
    "converters": {
        "variable": parse_list_of_strings,
    },
    "parse_dates": ["date_start", "date_end"],
    "date_parser": lambda s: pd.Period(s, "H"),
}


class DataCatalog(intake_esm.esm_datastore):
    """A read-only intake_esm catalog adapted to xscen's syntax."""

    def __init__(self, *args, check_valid=False, drop_duplicates=False, **kwargs):

        kwargs["read_csv_kwargs"] = recursive_update(
            csv_kwargs.copy(), kwargs.get("read_csv_kwargs", {})
        )
        super().__init__(*args, **kwargs)
        if check_valid:
            self.check_valid()
        if drop_duplicates:
            self.drop_duplicates()

    @classmethod
    def from_csv(
        cls,
        paths: Union[os.PathLike, Sequence[os.PathLike]],
        esmdata: Optional[Union[os.PathLike, dict]] = None,
        *,
        read_csv_kwargs: Mapping[str, Any] = None,
        name: str = "virtual",
        **intake_kwargs,
    ):
        """Create a DataCatalog from one or more csv files.

        Parameters
        ----------
        paths: paths or sequence of paths
          One or more paths to csv files.
        esmdata: path or dict, optional
          The "ESM collection data" as a path to a json file or a dict.
          If None (default), `catalog._esm_col_data` is used.
        read_csv_kwargs : dict, optional
          Extra kwargs to pass to `pd.read_csv`, in addition to the ones in `catalog.csv_kwargs`.
        name: str, optional
          If `metadata` doesn't contain it, a name to give to the catalog.
        """
        if isinstance(paths, os.PathLike):
            paths = [paths]

        if isinstance(esmdata, os.PathLike):
            with open(esmdata) as f:
                esmdata = json.load(f)
        elif esmdata is None:
            esmdata = deepcopy(_esm_col_data)
        if "id" not in esmdata:
            esmdata["id"] = name

        read_csv_kwargs = recursive_update(csv_kwargs.copy(), read_csv_kwargs or {})

        df = pd.concat([pd.read_csv(p, **read_csv_kwargs) for p in paths]).reset_index(
            drop=True
        )

        # Create the intake catalog
        return cls({"esmcat": esmdata, "df": df}, **intake_kwargs)

    def __dir__(self) -> List[str]:
        rv = ["iter_unique", "drop_duplicates", "check_valid"]
        return super().__dir__() + rv

    def unique(self, columns: Union[str, list] = None):
        """
        Simpler way to get unique elements from a column in the catalog.
        """
        if self.df.size == 0:
            raise ValueError("Catalog is empty.")
        out = super().unique()
        if columns is not None:
            out = out[columns]
        return out

    def iter_unique(self, *columns):
        """Iterate over sub-catalogs for each group of unique values for all specified columns.

        This is a generator that yields a tuple of the unique values of the current
        group, in the same order as the arguments, and the sub-catalog.
        """
        for values in itertools.product(*map(self.unique, columns)):
            sim = self.search(**dict(zip(columns, values)))
            if sim:  # So we never yield empty catalogs
                yield values, sim

    def drop_duplicates(self, columns=["id", "path"]):
        # In case variables are being added in an existing Zarr, append them
        if len(self.df) > 0:
            # By default, duplicated will mark the later entries as True
            duplicated = self.df.duplicated(subset="path")
            df_dupes = self.df[duplicated]
            for _, d in df_dupes.iterrows():
                if Path(d.path).suffix == ".zarr":
                    append_v = list()
                    [
                        append_v.extend(v)
                        for v in self.df[self.df["path"] == d["path"]]["variable"]
                    ]
                    # Since setting multiple entries to tuples is a pain, update only the duplicated and re-add it to df
                    # Other entries will be dropped by drop_duplicates
                    d["variable"] = tuple(set(append_v))
                    self.esmcat._df = pd.concat(
                        [self.esmcat._df, pd.DataFrame(d).transpose()]
                    )

        # Drop duplicates
        self.esmcat.df.drop_duplicates(
            subset=columns, keep="last", ignore_index=True, inplace=True
        )

    def check_valid(self):
        # In case files were deleted manually, double-check that files do exist
        def check_existing(row):
            path = Path(row.path)
            exists = (path.is_dir() and path.suffix == ".zarr") or (path.is_file())
            if not exists:
                logger.info(
                    f"File {path} was not found on disk, removing from catalog."
                )
            return exists

        # In case variables were deleted manually in a Zarr, double-check that they still exist
        def check_variables(row):
            path = Path(row.path)
            if path.suffix == ".zarr":
                variables = [p.parts[-1] for p in path.iterdir()]
                exists = tuple(
                    set(
                        [row.variable]
                        if isinstance(row.variable, str)
                        else row.variable
                    ).intersection(variables)
                )
            else:
                exists = row.variable
            return exists

        if len(self.df) > 0:
            self.esmcat._df = self.df[
                self.df.apply(check_existing, axis=1)
            ].reset_index(drop=True)
            self.esmcat._df["variable"] = self.df.apply(check_variables, axis=1)

    def exists_in_cat(self, **columns):
        """
        Check if there is an entry in the catalogue corresponding to the arguments given

        Parameters
        ----------
        cat: catalogue
        columns: Arguments that will be given to `catalog.search`

        Returns
        -------
        Boolean if an entry exist

        """
        exists = bool(len(self.search(**columns)))
        if exists:
            logger.info(f"An entry exists for: {columns}")
        return exists


class ProjectCatalog(DataCatalog):
    """A DataCatalog with additional 'write' functionalities that can update and upload itself."""

    @classmethod
    def create(
        cls,
        filename: Union[os.PathLike, str],
        *,
        project: Optional[dict] = None,
        overwrite=False,
    ):
        r"""Create a new project catalog from some project metadata.

        Creates the json from default `_esmcol_data` and an empty csv file.

        Parameters
        ----------
        filename : PathLike
          A path to the json file (with or without suffix).
        project : dict-like
          Metadata to create the catalog. If None, `CONFIG['project']` will be used.
          Valid fields are:

          - title : Name of the project, given as the catalog's "title".
          - id : slug-like version of the name, given as the catalog's id (should be url-proof)
                 Defaults to a modified name.
          - version : Version of the project (and thus the catalog), string like "x.y.z".
          - description : Detailed description of the project, given to the catalog's "description".
          - Any other entry defined in catalog._esm_col_data

          At least one of `id` and `title` must be given, the rest is optional.
        overwrite : bool
          If True, will overwrite any existing JSON and CSV file

        Returns
        -------
        ProjectCatalog
          An empty intake_esm catalog.
        """
        path = Path(filename)
        meta_path = path.with_suffix(".json")
        data_path = path.with_suffix(".csv")

        if (meta_path.is_file() or data_path.is_file()) and not overwrite:
            raise FileExistsError(
                "Catalog file already exist (at least one of {meta_path} or {data_path})."
            )

        meta_path.parent.mkdir(parents=True, exist_ok=True)

        project = project or CONFIG.get("project") or {}

        if "id" not in project and "title" not in project:
            raise ValueError(
                'At least one of "id" or "title" must be given in the metadata.'
            )

        project["catalog_file"] = str(data_path)
        if "id" not in project:
            project["id"] = project.get("title", "").replace(" ", "")

        esmdata = recursive_update(_esm_col_data.copy(), project)

        df = pd.DataFrame(columns=COLUMNS)

        cat = cls(
            {"esmcat": esmdata, "df": df}
        )  # TODO: Currently, this drops "version" because it is not a recognized attribute
        cat.serialize(
            path.stem,
            directory=path.parent,
            catalog_type="file",
            to_csv_kwargs={"compression": None},
        )

        # Change catalog_file to a relative path
        with open(meta_path) as f:
            meta = json.load(f)
            meta["catalog_file"] = data_path.name
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return cls(str(meta_path))

    def __init__(self, df, *args, **kwargs):
        super().__init__(df, *args, **kwargs)
        self.check_valid()
        self.drop_duplicates()
        self.meta_file = df if not isinstance(df, dict) else None

    # TODO: Implement a way to easily destroy part of the catalog to "reset" some steps
    def update(
        self,
        df: Optional[
            Union[
                "DataCatalog",
                intake_esm.esm_datastore,
                pd.DataFrame,
                pd.Series,
                Sequence[pd.Series],
            ]
        ] = None,
    ):
        """Updates the catalog with new data and writes the new data to the csv file.

        Once the internal dataframe is updated with `df`, the csv on disk is parsed,
        updated with the internal dataframe, duplicates are dropped and everything is
        written back to the csv. This means that nothing is _removed_* from the csv when
        calling this method, and it is safe to use even with a subset of the catalog.

        Warnings
        --------
        If a file was deleted between the parsing of the catalog and this call,
        it will be removed from the csv when `check_valid` is called.

        Parameters
        ----------
        df : Union[pd.DataFrame, pd.Series, DataCatalog]
          Data to be added to the catalog.
        """
        # Append the new DataFrame or Series
        if isinstance(df, DataCatalog) or isinstance(df, intake_esm.esm_datastore):
            self.esmcat._df = pd.concat([self.df, df.df])
        elif df is not None:
            if isinstance(df, pd.Series):
                df = pd.DataFrame(df).transpose()
            self.esmcat._df = pd.concat([self.df, df])

        self.check_valid()
        self.drop_duplicates()

        if self.meta_file is not None:
            with fs.open(self.esmcat.catalog_file, "wb") as csv_outfile:
                self.df.to_csv(csv_outfile, index=False, compression=None)
        else:
            # Update the catalog file saved on disk
            disk_cat = DataCatalog(
                {
                    "esmcat": self.esmcat.dict(),
                    "df": pd.read_csv(self.esmcat.catalog_file, **self.read_csv_kwargs),
                }
            )
            disk_cat.esmcat._df = pd.concat([disk_cat.df, self.df])
            disk_cat.check_valid()
            disk_cat.drop_duplicates()
            with fs.open(disk_cat.esmcat.catalog_file, "wb") as csv_outfile:
                disk_cat.df.to_csv(csv_outfile, index=False, compression=None)

    def update_from_ds(
        self,
        ds: xarray.Dataset,
        path: str,
        info_dict: Optional[dict] = None,
    ):
        """Updates the catalog with new data and writes the new data to the csv file.
        We get the new data from the attributes of `ds`, the dictionary `info_dict` and `path`.

        Once the internal dataframe is updated with the new data, the csv on disk is parsed,
        updated with the internal dataframe, duplicates are dropped and everything is
        written back to the csv. This means that nothing is _removed_* from the csv when
        calling this method, and it is safe to use even with a subset of the catalog.

        Warnings
        --------
        If a file was deleted between the parsing of the catalog and this call,
        it will be removed from the csv when `check_valid` is called.

        Parameters
        ----------
        ds : xarray.Dataset
          Dataset that we want to add to the catalog.
          The columns of the catalog will be filled from the global attributes starting with 'cat/' of the dataset.
        info_dict: dict
          Optional extra information to fill the catalog.
        path: str
          Path where ds is stored
        """
        d = {}

        for col in self.df.columns:
            if f"cat/{col}" in ds.attrs:
                d[col] = ds.attrs[f"cat/{col}"]
        if info_dict:
            d.update(info_dict)

        if "time" in ds:
            d["date_start"] = str(
                ds.isel(time=0).time.dt.strftime("%Y-%m-%d %H:%M:%S").values
            )
            d["date_end"] = str(
                ds.isel(time=-1).time.dt.strftime("%Y-%m-%d %H:%M:%S").values
            )

        d["path"] = path

        # variable should be based on the Dataset
        d["variable"] = tuple(v for v in ds.data_vars if len(ds[v].dims) > 0)

        if "format" not in d:
            d["format"] = Path(d["path"]).suffix.split(".")[1]
            logger.info(
                f"File format not specified. Adding it as '{d['format']}' based on file name."
            )

        self.update(pd.Series(d))

    def refresh(self):
        """
        Re-reads the catalog csv saved on disk.
        """
        if self.meta_file is None:
            raise ValueError(
                "Only full catalogs can be refreshed, but this instance is only a subset."
            )
        self.esmcat = ESMCatalogModel.load(
            self.meta_file, read_csv_kwargs=self.read_csv_kwargs
        )
        initlen = len(self.esmcat.df)
        self.check_valid()
        self.drop_duplicates()
        if len(self.df) != initlen:
            self.update()

    def __repr__(self) -> str:
        return (
            f'<{self.esmcat.id or ""} project catalog with {len(self)} dataset(s) from '
            f'{len(self.df)} asset(s) ({"subset" if self.meta_file is None else "full"})>'
        )


def concat_data_catalogs(*dcs):
    """Concatenate a multiple DataCatalogs.

    Output catalog is the union of all rows and all derived variables, with the the "esmcat"
    of the first DataCatalog. Duplicate rows are dropped and the index is reset.
    """
    registry = {}
    catalogs = []
    requested_variables = []
    requested_variables_true = []
    dependent_variables = []
    requested_variable_freqs = []
    for dc in dcs:
        registry.update(dc.derivedcat._registry)
        catalogs.append(dc.df)
        requested_variables.extend(dc._requested_variables)
        requested_variables_true.extend(dc._requested_variables_true)
        dependent_variables.extend(dc._dependent_variables)
        requested_variable_freqs.extend(dc._requested_variable_freqs)
    df = pd.concat(catalogs, axis=0).drop_duplicates(ignore_index=True)
    dvr = intake_esm.DerivedVariableRegistry()
    dvr._registry.update(registry)
    newcat = DataCatalog({"esmcat": dcs[0].esmcat.dict(), "df": df}, registry=dvr)
    newcat._requested_variables = requested_variables
    newcat._requested_variables_true = requested_variables_true
    newcat._dependent_variables = dependent_variables
    newcat._requested_variable_freqs = requested_variable_freqs
    return newcat


@parse_config
def get_asset_list(root_paths, globpat="*.nc", parallel_depth=1):
    """List files fitting a given glob pattern from a list of paths.

    Search is done with GNU's `find` and parallized through `dask`.
    """
    if isinstance(root_paths, str):
        root_paths = [root_paths]

    # Support for wildcards in root_paths
    root_star = deepcopy(root_paths)
    for r in root_star:
        if "*" in r:
            new_roots = [
                x
                for x in glob(str(r))
                if Path(x).is_dir() and Path(x).suffix != ".zarr"
            ]
            root_paths.remove(r)
            root_paths.extend(new_roots)

    @dask.delayed
    def _file_dir_files(directory, pattern):
        try:
            if "/" in pattern:
                *foldparts, extension = pattern.split("/")
                foldpatt = "/".join(foldparts)
                folders = subprocess.run(
                    [
                        "find",
                        "-L",
                        directory.as_posix(),
                        "-type",
                        "d",
                        "-wholename",
                        foldpatt,
                    ],
                    capture_output=True,
                    encoding="utf-8",
                ).stdout.split()
                output = []
                for folder in folders:
                    output.extend(
                        subprocess.run(
                            ["find", "-L", folder, "-name", extension],
                            capture_output=True,
                            encoding="utf-8",
                        ).stdout.split()
                    )
            else:
                output = subprocess.run(
                    ["find", "-L", directory.as_posix(), "-name", pattern],
                    capture_output=True,
                    encoding="utf-8",
                ).stdout.split()
        except Exception:
            output = []
        return output

    filecoll = dict()
    for r in root_paths:
        root = Path(r)
        dirs = []
        for d in range(1, parallel_depth + 1):
            pattern = "*/" * d
            for x in root.glob(pattern):
                if x.is_dir() and x.suffix != ".zarr" and x.parent.suffix != ".zarr":
                    if x.parent in dirs:
                        dirs.remove(x.parent)
                    dirs.append(x)

        filelist = [_file_dir_files(directory, globpat) for directory in dirs]
        # watch progress
        with ProgressBar():
            filelist = dask.compute(*filelist)
        filecoll[root] = list(itertools.chain(*filelist))

        # add files in the first directory, unless the glob pattern includes
        # folder matching, in which case we don't want to risk recomputing the same parsing.
        if "/" not in globpat:
            filecoll[root].extend([str(x.name) for x in root.glob(f"{globpat}")])
        filecoll[root].sort()

    return filecoll


def name_parser(
    path, root, patterns, read_from_file=None, attrs_map=None, xr_open_kwargs=None
):
    """Extract metadata information from the file path.

    Parameters
    ----------
    path : str
      Full file path.
    patterns : list or str
      List of patterns to try in `reverse_format`
    read_from_file : list of string or dict, optional
      If not None, passed directly to :py:func:`parse_from_ds` as `names`.
    attrs_map:
      If `read_from_file` is not None, passed directly to :py:func:`parse_from_ds`.
    xr_open_kwargs: dict
      If required, arguments to send to xr.open_dataset() when opening the file to read the attributes.

    Returns
    -------
    dict
        The metadata fields parsed from the path using the first fitting pattern.
        If no pattern matched, {'path': None} is returned.
    """
    abs_path = Path(path)
    path = abs_path.relative_to(root)
    xr_open_kwargs = xr_open_kwargs or {}

    d = {}
    for pattern in patterns:
        if len(Path(pattern).parts) != len(path.parts):
            continue
        try:
            d = reverse_format(pattern, str(path))
            fields_with_illegal = {
                k: v
                for k, v in d.items()
                if ("_" in v and not k.startswith("*")) or "/" in v
            }
            if d and not fields_with_illegal:
                break
            else:
                d = {}
        except ValueError:
            continue
    if not d:
        logger.debug(f"No pattern matched with path {path}")
        return {"path": None}

    logger.debug(f"Parsed file path {path} and got {len(d)} fields.")

    # files with a single year/month
    if "date_end" not in d and "date_start" in d:
        d["date_end"] = d["date_start"]

    d["path"] = abs_path
    d["format"] = path.suffix[1:]

    if read_from_file:
        try:
            fromfile = parse_from_ds(
                abs_path, names=read_from_file, attrs_map=attrs_map, **xr_open_kwargs
            )
        except Exception as err:
            logger.error(f"Unable to parse file {path}, got : {err}")
        else:
            d.update(fromfile)
    # strip to clean off lost spaces and line jumps
    return {
        k: v.strip() if isinstance(v, str) else v
        for k, v in d.items()
        if not k.startswith("*") and not k.startswith("?")  # Wildcards
    }


@parse_config
def parse_directory(
    directories: list,
    globpattern: str,
    patterns: list,
    *,
    id_columns: list = None,
    read_from_file: Union[
        bool,
        Sequence[str],
        Tuple[Sequence[str], Sequence[str]],
        Sequence[Tuple[Sequence[str], Sequence[str]]],
    ] = False,
    homogenous_info: dict = None,
    cvs: Union[str, PosixPath, dict] = None,
    xr_open_kwargs: Mapping[str, Any] = None,
    parallel_depth: int = 1,
    only_official_columns: bool = True,
) -> pd.DataFrame:
    r"""Parse files in a directory and return them as a pd.DataFrame.

    Parameters
    ----------
    directories : list
        List of directories to parse. The parse is recursive and accepts wildcards (*).
    globpattern: str
        A glob pattern for file name matching, usually only a suffix like "*.nc".
        May include folder matching, in which case don't forget that the search is parallelized for
        subfolders up to the depth given by `parallel_depth`. Also, contrary to real posix glob patterns,
        this makes no difference between "**" and "*".
    patterns : list
        List of possible patterns to be used by intake.source.utils.reverse_filename() to decode the file names. See Notes below.
    id_columns : list
        List of column names on which to base the dataset definition. Empty columns will be skipped.
    read_from_file : boolean or set of strings or tuple of 2 sets of strings.
        If True, if some fields were not parsed from their path, files are opened and
        missing fields are parsed from their metadata, if found.
        If a set of column names, only those fields are parsed from the file, if missing.
        If False (default), files are never opened.
        If a tuple of 2 lists of strings, only the first file of groups defined by the
        first list of columns is read and the second list of columns is parsed from the
        file and applied to the whole group.
        It can also be a list of those tuples.
    homogenous_info : dict, optional
        Using the {column_name: description} format, information to apply to all files.
    cvs: str or PosixPath or dict, optional
        Dictionary with mapping from parsed name to controlled names for each column.
        May have an additionnal "attributes" entry which maps from attribute names in the files to
        official column names. The attribute translation is done before the rest.
    xr_open_kwargs: dict
        If needed, arguments to send xr.open_dataset() when opening the file to read the attributes.
    parallel_depth: int
        The depth at which to parallelize the finding of files in the directories. A value of 1 (default and minimum),
        means each subfolder of a given directory are searched in parallel.
    only_official_columns: bool
        If True (default), this ensure the final catalog has all official columns and only those. Other fields in the patterns will raise an error.
        If False, the columns are those used in the patterns and the homogenous info. In that case, the column order is not determined.
        Path, format and id are always present in the output.

    Notes
    -----
    - Offical columns names are: ["id", "type", "processing_level", "mip_era", "activity", "driving_institution", "driving_model", "institution",
                          "source", "bias_adjust_institution", "bias_adjust_project","experiment", "member",
                          "xrfreq", "frequency", "variable", "domain", "date_start", "date_end", "version"]
    - Not all column names have to be present, but "xrfreq" (obtainable through "frequency"), "variable",
        "date_start" and "processing_level" are necessary for a workable catalog.
    - 'patterns' should highlight the columns with braces.
        Wildcards can be used for irrelevant parts of a path. "*" means anything including the "_" character (but not "/"),
        while "?" fits any string without "_". Any character following the wildcard is ignored, in order to have human readable placeholders.
        Example: `"{?ignored project name}_{?}_{domain}_{?}_{variable}_{date_start}_{activity}_{experiment}_{processing_level}_{*gibberish}.nc"`

    Returns
    -------
    pd.DataFrame
        Parsed directory files
    """
    homogenous_info = homogenous_info or {}
    xr_open_kwargs = xr_open_kwargs or {}
    pattern_fields = set.union(
        *map(lambda p: set(re.findall(r"{([\w]*)}", p)), patterns)
    )
    if only_official_columns:
        columns = set(COLUMNS) - homogenous_info.keys()
        unrecognized = pattern_fields - set(COLUMNS)
        if unrecognized:
            raise ValueError(
                f"Patterns include fields which are not recognized by xscen : {unrecognized}. "
                "If this is wanted, pass only_official_columns=False to remove the check."
            )
    else:
        # Get all columns defined in the patterns
        columns = pattern_fields.union({"path", "format"})

    read_file_groups = False  # Whether to read file per groupe or not.
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

    # Find files
    filecoll = get_asset_list(
        directories, globpat=globpattern, parallel_depth=parallel_depth
    )
    logger.info(
        f"Found {sum(len(filelist) for filelist in filecoll.values())} files to parse."
    )

    # Parse the file names
    def _update_dict(entry):
        z = {k: entry.get(k) for k in columns}
        return z

    @dask.delayed
    def delayed_parser(*args, **kwargs):
        return _update_dict(name_parser(*args, **kwargs))

    parsed = [
        delayed_parser(
            x,
            root,
            patterns,
            read_from_file=read_from_file if not read_file_groups else None,
            attrs_map=attrs_map,
            xr_open_kwargs=xr_open_kwargs,
        )
        for root, filelist in filecoll.items()
        for x in filelist
    ]
    if len(parsed) == 0:
        raise FileNotFoundError("No files found while parsing.")

    with ProgressBar():
        parsed = dask.compute(*parsed)
    # Path has become NaN when some paths didn't fit any passed pattern
    df = pd.DataFrame(parsed).dropna(axis=0, subset=["path"])

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
        # Read all CVs and replace values in catalog accordingly
        df = df.replace(cvs)

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
    # todo
    # 1. Les dates NaT correspondent à une fréquence fx
    # 2. Toutes les xrfreq sont correctes (pandas + fx)
    # 3. Format est zarr ou nc

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
        ds_attrs, variables = _parse_from_zarr(obj, get_vars="variables" in names)
        time = None
    elif isinstance(obj, Path) and obj.suffixes[-1] == ".nc":
        logger.info(f"Parsing attributes with netCDF4 from {obj}.")
        ds_attrs, variables, time = _parse_from_nc(
            obj, get_vars="variables" in names, get_time=get_time
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
            attrs["variable"] = tuple(variables)
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
        elif name in ds_attrs:
            attrs[name] = ds_attrs[name].strip()
        elif name in rev_attrs_map and rev_attrs_map[name] in ds_attrs:
            attrs[name] = ds_attrs[rev_attrs_map[name]].strip()

    logger.debug(f"Got fields {attrs.keys()} from file.")
    return attrs


def _parse_from_zarr(path: os.PathLike, get_vars=True):
    """Obtains the list of variables and the list of global attributes from a zarr dataset, reading the json files directly.

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
            varpath.name for varpath in path.iterdir() if varpath.name not in coords
        ]
    return ds_attrs, variables


def _parse_from_nc(path: os.PathLike, get_vars=True, get_time=True):
    """Obtains the list of variables, the time coordinate and the list of global attributes from a netCDF dataset, using netCDF4.

    Variabl
    """
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


def date_parser(
    date,
    *,
    end_of_period: bool = False,
    out_dtype: str = "period",
    strtime_format: str = "%Y-%m-%d",
    freq: str = "H",
) -> Union[str, pd.Period, pd.Timestamp]:
    """
    Returns a datetime from a string

    Parameters
    ----------
    date : str
      Date to be converted
    end_of_period : bool, optional
      If True, the date will be the end of month or year depending on what's most appropriate
    out_dtype: str, optional
      Choices are 'period', 'datetime' or 'str'
    strtime_format: str, optional
      If out_dtype=='str', this sets the strftime format
    freq : str
      If out_dtype=='period', this sets the frequency of the period.

    Returns
    -------
    pd.Period, pd.Timestamp, str
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
        19: ["%Y-%m-%dT%H:%M:%S"],
    }

    def _parse_date(date, fmts):
        for fmt in fmts:
            try:
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
    if isinstance(date, str):
        try:
            date, fmt = _parse_date(date, fmts[len(date)])
        except (KeyError, ValueError):
            date = pd.NaT
    elif isinstance(date, cftime.datetime):
        for n in range(3):
            try:
                date = datetime.fromisoformat((date - pd.Timedelta(n)).isoformat())
            except ValueError:  # We are NOT catching OutOfBoundsDatetime.
                pass
            else:
                break
        else:
            raise ValueError(
                "Unable to parse cftime date {date}, even when moving back 2 days."
            )
    elif isinstance(date, pd.Timestamp):
        date = date.to_pydatetime()

    if not isinstance(date, pd.Period):
        date = pd.Period(date, freq=freq)

    if end_of_period and fmt:
        if "m" not in fmt:
            date = date.asfreq("A-DEC").asfreq(freq)
        elif "d" not in fmt:
            date = date.asfreq("M").asfreq(freq)
        # TODO: Implement subdaily ?

    if out_dtype == "str":
        return date.strftime(strtime_format)

    if out_dtype == "datetime":
        return date.to_timestamp()
    return date


def generate_id(df: pd.DataFrame, id_columns: Optional[list] = None):
    """Utility to create an ID from column entries.

    Parameters
    ----------
    df: pd.DataFrame
      Data for which to create an ID.
    id_columns : list
      List of column names on which to base the dataset definition. Empty columns will be skipped.
    """

    id_columns = [x for x in (id_columns or ID_COLUMNS) if x in df.columns]

    return df[id_columns].apply(
        lambda row: "_".join(map(str, filter(pd.notna, row.values))), axis=1
    )
