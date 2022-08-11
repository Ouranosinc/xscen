import ast
import itertools
import json
import logging
import os
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
import pandas as pd
import xarray
import xarray as xr
from dask.diagnostics import ProgressBar
from intake.source.utils import reverse_format
from intake_esm.cat import ESMCatalogModel

from .config import CONFIG, parse_config, recursive_update
from .io import get_engine
from .utils import CV

logger = logging.getLogger(__name__)
# Monkey patch for attribute names in the output of to_dataset_dict
intake_esm.set_options(attrs_prefix="cat")

__all__ = [
    "COLUMNS",
    "ID_COLUMNS",
    "concat_data_catalogs",
    "date_parser",
    "generate_id",
    "parse_directory",
    "parse_from_ds",
]


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


def _parse_list_of_strings(elem):
    """Parse an element of a csv in case it is a list of strings."""
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
        "variable": _parse_list_of_strings,
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

    def drop_duplicates(self, columns: Optional[List[str]] = None):
        # In case variables are being added in an existing Zarr, append them
        if columns is None:
            columns = ["id", "path"]

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

        for col in COLUMNS:
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
def _get_asset_list(root_paths, extension="*.nc"):
    """List files with a given extension from a list of paths.

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
    def _file_dir_files(directory, extension):
        try:
            cmd = ["find", "-L", directory.as_posix(), "-name", extension]
            proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            output = proc.stdout.read().decode("utf-8").split()
        except Exception:
            output = []
        return output

    filelist = list()
    for r in root_paths:
        root = Path(r)
        pattern = "*/"

        dirs = [x for x in root.glob(pattern) if x.is_dir() and x.suffix != ".zarr"]

        filelistroot = [_file_dir_files(directory, extension) for directory in dirs]
        # watch progress
        with ProgressBar():
            filelistroot = dask.compute(*filelistroot)
        filelist.extend(list(itertools.chain(*filelistroot)))

        # add files in the first directory
        filelist.extend([str(x) for x in root.glob(f"{extension}")])

    return sorted(filelist)


def _name_parser(
    path, patterns, read_from_file=None, xr_open_kwargs: Mapping[str, Any] = None
):
    """Extract metadata information from the file path.

    Parameters
    ----------
    path : str
      Full file path.
    patterns : list or str
      List of patterns to try in `reverse_format`
    read_from_file : list of string, optional
      A list of columns to parse from the file's metadata itself.
    xr_open_kwargs: dict
        If required, arguments to send xr.open_dataset() when opening the file to read the attributes.
    """
    path = Path(path)
    xr_open_kwargs = xr_open_kwargs or {}

    d = {}
    for pattern in patterns:
        folder_depth = len(Path(pattern).parts) - 1
        # stem is a path with the same number of parts as the pattern
        stem = str(Path(*path.parts[-1 - folder_depth :]))
        try:
            d = reverse_format(pattern, stem)
            if d:
                break
        except ValueError:
            continue
    if not d:
        logger.warn(f"No pattern matched with path {path}..")
    else:
        logger.debug(f"Parsed file path {path} and got {len(d)} fields.")

    # files with a single year/month
    if ("date_end" not in d.keys()) and ("date_start" in d.keys()):
        d["date_end"] = d["date_start"]

    d["path"] = path
    d["format"] = path.suffix[1:]

    if read_from_file:
        missing = set(read_from_file) - d.keys()
        if missing:
            try:
                fromfile = parse_from_ds(path, names=missing, **xr_open_kwargs)
            except Exception as err:
                logger.error(f"Unable to parse file {path}, got : {err}")
            finally:
                d.update(fromfile)
                logger.debug(
                    f"Parsed file data and got {len(missing.intersection(d.keys()))} more fields."
                )
    # strip to clean off lost spaces and line jumps
    return {k: v.strip() if isinstance(v, str) else v for k, v in d.items()}


@parse_config
def parse_directory(
    directories: list,
    extension: str,
    patterns: list,
    *,
    id_columns: list = None,
    read_from_file: Union[
        bool, Sequence[str], Tuple[Sequence[str], Sequence[str]]
    ] = False,
    homogenous_info: dict = None,
    cvs_dir: Union[str, PosixPath] = None,
    xr_open_kwargs: Mapping[str, Any] = None,
) -> pd.DataFrame:
    r"""Parse files in a directory and return them as a pd.DataFrame.

    Parameters
    ----------
    directories : list
        List of directories to parse. The parse is recursive and accepts wildcards (*).
    extension: str
        A glob pattern for file name matching, usually only a suffix like ".nc".
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
    homogenous_info : dict, optional
        Using the {column_name: description} format, information to apply to all files.
    cvs_dir: Union[str, PosixPath], optional
        Directory where JSON controlled vocabulary files are located. See Notes below.
    xr_open_kwargs: dict
        If required, arguments to send xr.open_dataset() when opening the file to read the attributes.

    Notes
    -----
    - Columns names are: ["id", "type", "processing_level", "mip_era", "activity", "driving_institution", "driving_model", "institution",
                          "source", "bias_adjust_institution", "bias_adjust_project","experiment", "member",
                          "xrfreq", "frequency", "variable", "domain", "date_start", "date_end", "version"]
    - Not all column names have to be present, but "xrfreq", "variable", "date_start", "date_end" & "processing_level" are necessary.
    - 'patterns' should highlight the columns with braces.
        - A wildcard can be used for irrelevant parts of a filename.
        - Example: "{*}_{*}_{domain}_{*}_{variable}_{date_start}_{activity}_{experiment}_{processing_level}_{*}.nc"
    - JSON files for the controlled vocabulary must have the same name as the column. One file per column.

    Returns
    -------
    pd.DataFrame
      Parsed directory files

    """
    homogenous_info = homogenous_info or {}
    columns = set(COLUMNS) - homogenous_info.keys()
    first_file_only = (
        None  # The set of columns defining groups for which read the first file.
    )
    if not isinstance(read_from_file, bool) and not isinstance(read_from_file[0], str):
        # A tuple of 2 lists
        first_file_only, read_from_file = read_from_file
    if read_from_file is True:
        # True but not a list of strings
        read_from_file = columns
    elif read_from_file is False:
        read_from_file = set()

    filelist = _get_asset_list(directories, extension=extension)
    logger.info(f"Found {len(filelist)} files to parse.")

    def _update_dict(entry):
        z = {k: entry.get(k) for k in columns}
        return z

    @dask.delayed
    def delayed_parser(*args, **kwargs):
        return _update_dict(_name_parser(*args, **kwargs))

    parsed = [
        delayed_parser(
            x,
            patterns,
            read_from_file=read_from_file if first_file_only is None else [],
            xr_open_kwargs=xr_open_kwargs if first_file_only is None else {},
        )
        for x in filelist
    ]
    with ProgressBar():
        parsed = dask.compute(*parsed)
    df = pd.DataFrame(parsed)

    def read_first_file(grp, cols, xrkwargs):
        fromfile = parse_from_ds(grp.path.iloc[0], cols, **xrkwargs)
        logger.info(f"Got {len(fromfile)} fields, applying to {len(grp)} entries.")
        out = grp.copy()
        for col, val in fromfile.items():
            for i in grp.index:  # If val is an iterable we can't use loc.
                out.at[i, col] = val
        return out

    if first_file_only is not None:
        df = (
            df.groupby(first_file_only)
            .apply(
                read_first_file, cols=read_from_file, xrkwargs=(xr_open_kwargs or {})
            )
            .reset_index(drop=True)
        )

    if df.shape[0] == 0:
        raise FileNotFoundError("No files found while parsing.")

    # add homogeous info
    for key, val in homogenous_info.items():
        df[key] = val

    # Replace DataFrame entries by definitions found in CV
    if cvs_dir is not None:
        # Read all CVs and replace values in catalog accordingly
        cvs = {}
        for cvpath in Path(cvs_dir).glob("*.json"):
            if cvpath.stem in df.columns:
                with cvpath.open("r") as f:
                    cvs[cvpath.stem] = json.load(f)
        df = df.replace(cvs)

    # Parse dates
    df["date_start"] = df["date_start"].apply(date_parser)
    df["date_end"] = df["date_end"].apply(date_parser, end_of_period=True)

    # translate xrfreq into frequencies and vice-versa
    df["xrfreq"].fillna(
        df["frequency"].apply(CV.frequency_to_xrfreq, default=pd.NA), inplace=True
    )
    df["frequency"].fillna(
        df["xrfreq"].apply(CV.xrfreq_to_frequency, default=pd.NA), inplace=True
    )

    # Create id from user specifications
    df["id"] = generate_id(df, id_columns)

    # ensure path is a string
    df["path"] = df.path.apply(str)

    # Sort columns and return
    return df.loc[:, COLUMNS]


def parse_from_ds(
    obj: Union[os.PathLike, xr.Dataset], names: Sequence[str], **xrkwargs
):
    """Parse a list of catalog fields from the file/dataset itself.

    If passed a path, this opens the file.

    Infers the variable from the variables.
    Infers frequency, date_start and date_end from the time coordinate if present.
    Infers other attributes from the coordinates or the global attributes.
    """
    attrs = {}
    if not isinstance(obj, xr.Dataset):
        ds = xr.open_dataset(obj, engine=get_engine(obj), **xrkwargs)
        logger.info(f"Parsing attributes from file {obj}.")
    else:
        ds = obj
        logger.info("Parsing attributes from dataset.")

    for name in names:
        if name == "variable":
            attrs["variable"] = tuple(
                set(ds.data_vars.keys()).difference(
                    [v for v in ds.data_vars if len(ds[v].dims) == 0]
                )
            )
        elif (
            name in ("frequency", "xrfreq")
            and name not in attrs
            and "time" in ds.coords
            and ds.time.size > 3
        ):
            # round to the minute to catch floating point imprecision
            freq = xr.infer_freq(ds.time.dt.round("T"))
            if freq:
                if "xrfreq" in names:
                    attrs["xrfreq"] = freq
                if "frequency" in names:
                    attrs["frequency"] = CV.xrfreq_to_frequency(freq)
            else:
                warnings.warn(
                    f"Couldn't infer frequency of dataset {obj if not isinstance(obj, xr.Dataset) else ''}"
                )
        elif name == "xrfreq" and "time" not in ds.coords:
            attrs["xrfreq"] = "fx"
        elif name == "date_start" and "time" in ds.coords:
            attrs["date_start"] = ds.indexes["time"][0]
        elif name == "date_end" and "time" in ds.coords:
            attrs["date_end"] = ds.indexes["time"][-1]
        elif name in ds.coords:
            attrs[name] = tuple(ds.coords[name].values)
        elif name in ds.attrs:
            attrs[name] = ds.attrs[name].strip()

    logger.debug(f"Got fields {attrs.keys()} from file.")
    return attrs


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
        date, fmt = _parse_date(date, fmts[len(date)])
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
