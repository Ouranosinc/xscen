"""Catalog objects and related tools."""

import ast
import itertools
import json
import logging
import os
import re
import shutil as sh
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import reduce
from operator import or_
from pathlib import Path
from typing import Any

import fsspec as fs
import intake_esm
import pandas as pd
import tlz
import xarray as xr
from intake_esm.cat import ESMCatalogModel

from .config import CONFIG, args_as_str, recursive_update
from .utils import (
    _xarray_defaults,
    date_parser,
    ensure_correct_time,
    ensure_new_xrfreq,
    standardize_periods,
)


logger = logging.getLogger(__name__)
# Monkey patch for attribute names in the output of to_dataset_dict
intake_esm.set_options(attrs_prefix="cat")


__all__ = [
    "COLUMNS",
    "ID_COLUMNS",
    "DataCatalog",
    "ProjectCatalog",
    "concat_data_catalogs",
    "generate_id",
    "subset_file_coverage",
    "unstack_id",
]


# As much as possible, these catalog columns and entries should align with:
# https://github.com/WCRP-CMIP/CMIP6_CVs and https://github.com/ES-DOC/pyessv-archive
# See docs/columns.rst for a description of each entry.
COLUMNS = [
    "id",
    "type",
    "processing_level",
    "bias_adjust_institution",
    "bias_adjust_project",
    "bias_adjust_reference",
    "mip_era",
    "activity",
    "driving_model",
    "driving_member",
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
"""Official column names."""

ID_COLUMNS = [
    "bias_adjust_project",
    "bias_adjust_reference",
    "mip_era",
    "activity",
    "driving_model",
    "driving_member",
    "institution",
    "source",
    "experiment",
    "member",
    "domain",
]
"""Default columns used to create a unique ID"""


esm_col_data = {
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
"""Default ESM column data for the official catalogs."""


def _parse_list_of_strings(elem):
    """Parse an element of a csv in case it is a tuple of strings."""
    if elem.startswith("(") or elem.startswith("["):
        out = ast.literal_eval(elem)
        return tuple(out)
    return (elem,)


def _parse_dates(elem):
    """Parse an array of dates (strings) into a PeriodIndex of hourly frequency."""
    # Cast to normal datetime as this is much faster than to period for in-bounds dates
    # errors are coerced to NaT, we convert to a PeriodIndex and then to a (mutable) series
    time = pd.to_datetime(elem, errors="coerce").astype(pd.PeriodDtype("H")).to_series()
    nat = time.isnull()
    # Only where we have NaT (parser errors and empty fields), parse into a Period
    # This will raise DateParseError as expected if the string is not parsable.
    time[nat] = pd.PeriodIndex(elem[nat], freq="H")
    return pd.PeriodIndex(time)


csv_kwargs = {
    "dtype": {
        key: "category" if not key == "path" else "string[pyarrow]" for key in COLUMNS if key not in ["xrfreq", "variable", "date_start", "date_end"]
    },
    "converters": {
        "variable": _parse_list_of_strings,
        "xrfreq": ensure_new_xrfreq,
    },
}
"""Kwargs to pass to `pd.read_csv` when opening an official Ouranos catalog."""


class DataCatalog(intake_esm.esm_datastore):
    r"""
    A read-only intake_esm catalog adapted to xscen's syntax.

    This class expects the catalog to have the columns listed in :py:data:`xscen.catalog.COLUMNS`
    and it comes with default arguments for reading the CSV files (:py:data:`xscen.catalog.csv_kwargs`).
    For example, all string columns (except `path`) are cast to a categorical dtype and the
    datetime columns are parsed with a special function that allows dates outside the conventional
    `datetime64[ns]` bounds by storing the data using :py:class:`pandas.Period` objects.

    Parameters
    ----------
    \*args : str or os.PathLike or dict
        Path to a catalog JSON file. If a dict, it must have two keys: 'esmcat' and 'df'.
        'esmcat' must be a dict representation of the ESM catalog.
        'df' must be a Pandas DataFrame containing content that would otherwise be in the CSV file.
    check_valid : bool
        If True, will check that all files in the catalog exist on disk and remove those that don't.
    drop_duplicates : bool
        If True, will drop duplicates in the catalog based on the 'id' and 'path' columns.
    \**kwargs : dict
        Any other arguments are passed to intake_esm.esm_datastore.

    See Also
    --------
    intake_esm.core.esm_datastore
    """

    def __init__(self, *args, check_valid: bool = False, drop_duplicates: bool = False, **kwargs):
        kwargs["read_csv_kwargs"] = recursive_update(csv_kwargs.copy(), kwargs.get("read_csv_kwargs", {}))
        args = args_as_str(args)

        super().__init__(*args, **kwargs)

        # Cast date columns into datetime (with ms reso, that's why we do it here and not in the `read_csv_kwargs`)
        # Pandas >=2 supports [ms] resolution, but can't parse strings with this resolution, so we need to go through numpy
        for datecol in ["date_start", "date_end"]:
            if datecol in self.df.columns and self.df[datecol].dtype == "O":
                # Missing values in object columns are np.nan, which numpy can't convert to datetime64 (what's up with that numpy???)
                self.df[datecol] = self.df[datecol].dropna().astype("datetime64[ms]")

        if check_valid:
            self.check_valid()
        if drop_duplicates:
            self.drop_duplicates()

    @classmethod
    def from_df(
        cls,
        data: pd.DataFrame | os.PathLike | Sequence[os.PathLike],
        esmdata: os.PathLike | dict | None = None,
        *,
        read_csv_kwargs: Mapping[str, Any] | None = None,
        name: str = "virtual",
        **intake_kwargs,
    ):
        """
        Create a DataCatalog from one or more csv files.

        Parameters
        ----------
        data: DataFrame or path or sequence of paths
          A DataFrame or one or more paths to csv files.
        esmdata: path or dict, optional
          The "ESM collection data" as a path to a json file or a dict.
          If None (default), xscen's default :py:data:`esm_col_data` is used.
        read_csv_kwargs : dict, optional
          Extra kwargs to pass to `pd.read_csv`, in addition to the ones in :py:data:`csv_kwargs`.
        name: str
          If `metadata` doesn't contain it, a name to give to the catalog.

        See Also
        --------
        pandas.read_csv
        """
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            if isinstance(data, os.PathLike):
                data = [data]

            read_csv_kwargs = recursive_update(csv_kwargs.copy(), read_csv_kwargs or {})

            df = pd.concat([pd.read_csv(pth, **read_csv_kwargs) for pth in data]).reset_index(drop=True)

        if isinstance(esmdata, os.PathLike):
            with Path(esmdata).open(encoding="utf-8") as f:
                esmdata = json.load(f)
        elif esmdata is None:
            esmdata = deepcopy(esm_col_data)
        if "id" not in esmdata:
            esmdata["id"] = name

        # Create the intake catalog
        return cls({"esmcat": esmdata, "df": df}, **intake_kwargs)

    def __dir__(self) -> list[str]:  # noqa: D105
        rv = ["iter_unique", "drop_duplicates", "check_valid"]
        return super().__dir__() + rv

    def _unique(self, columns) -> dict:
        def _find_unique(series):
            values = series.dropna()
            if series.name in self.esmcat.columns_with_iterables:
                values = tlz.concat(values)
            return list(tlz.unique(values))

        data = self.df[columns]
        if data.empty:
            return {col: [] for col in self.df.columns}
        else:
            return data.apply(_find_unique, result_type="reduce").to_dict()

    def unique(self, columns: str | Sequence[str] | None = None):
        """
        Return a series of unique values in the catalog.

        Parameters
        ----------
        columns : str or sequence of str, optional
          The columns to get unique values from. If None, all columns are used.
        """
        if self.df.size == 0:
            raise ValueError("Catalog is empty.")
        if isinstance(columns, str):
            cols = [columns]
        elif columns is None:
            cols = list(self.df.columns)
        else:
            cols = list(columns)
        uni = pd.Series(self._unique(cols))
        if isinstance(columns, str):
            return uni[columns]
        return uni

    def iter_unique(self, *columns):
        """
        Iterate over sub-catalogs for each group of unique values for all specified columns.

        This is a generator that yields a tuple of the unique values of the current
        group, in the same order as the arguments, and the sub-catalog.
        """
        for values in itertools.product(*self.unique(columns)):
            sim = self.search(**dict(zip(columns, values, strict=False)))
            if sim:  # So we never yield empty catalogs
                yield values, sim

    def search(self, **columns):
        """Modification of .search() to add the 'periods' keyword."""
        periods = columns.pop("periods", False)
        if len(columns) > 0:
            cat = super().search(**columns)
        else:
            cat = self.__class__({"esmcat": self.esmcat.dict(), "df": self.esmcat._df})
        if periods is not False:
            cat.esmcat._df = subset_file_coverage(cat.esmcat._df, periods=periods, coverage=0, duplicates_ok=True)
        return cat

    def drop_duplicates(self, columns: list[str] | None = None):
        """
        Drop duplicates in the catalog based on a subset of columns.

        Parameters
        ----------
        columns: list of str, optional
            The columns used to identify duplicates. If None, 'id' and 'path' are used.
        """
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
                    [append_v.extend(v) for v in self.df[self.df["path"] == d["path"]]["variable"]]
                    # Since setting multiple entries to tuples is a pain, update only the duplicated and re-add it to df
                    # Other entries will be dropped by drop_duplicates
                    d["variable"] = tuple(set(append_v))
                    self.esmcat._df = pd.concat([self.esmcat._df, pd.DataFrame(d).transpose()])

        # Drop duplicates
        self.esmcat.df.drop_duplicates(subset=columns, keep="last", ignore_index=True, inplace=True)

    def check_valid(self):
        """
        Verify that all files in the catalog exist on disk and remove those that don't.

        If a file is a Zarr, it will also check that all variables are present and remove those that aren't.
        """
        len_df = len(self.df)  # This line is required to avoid a D202 pydocstyle error

        # In case files were deleted manually, double-check that files do exist
        def check_existing(row):
            path = Path(row.path)
            exists = (path.is_dir() and path.suffix == ".zarr") or (path.is_file())
            if not exists:
                msg = f"File {path} was not found on disk, removing from catalog."
                logger.info(msg)
            return exists

        # In case variables were deleted manually in a Zarr, double-check that they still exist
        def check_variables(row):
            path = Path(row.path)
            if path.suffix == ".zarr":
                variables = [p.parts[-1] for p in path.iterdir()]
                exists = tuple(set([row.variable] if isinstance(row.variable, str) else row.variable).intersection(variables))
            else:
                exists = row.variable
            return exists

        if len_df > 0:
            self.esmcat._df = self.df[self.df.apply(check_existing, axis=1)].reset_index(drop=True)
            if len_df > 0:
                self.esmcat._df["variable"] = self.df.apply(check_variables, axis=1)

    def exists_in_cat(self, **columns) -> bool:
        """
        Check if there is an entry in the catalogue corresponding to the arguments given.

        Parameters
        ----------
        columns: Arguments that will be given to `catalog.search`

        Returns
        -------
        bool
            True if there is an entry in the catalogue corresponding to the arguments given.
        """
        exists = bool(len(self.search(**columns)))
        if exists:
            msg = f"An entry exists for: {columns}"
            logger.info(msg)
        return exists

    def to_dataset(
        self,
        concat_on: list[str] | str | None = None,
        create_ensemble_on: list[str] | str | None = None,
        ensemble_name: list[str] | None = None,
        calendar: str | None = "standard",
        **kwargs,
    ) -> xr.Dataset:
        """
        Open the catalog's entries into a single dataset.

        Same as :py:meth:`~intake_esm.core.esm_datastore.to_dask`, but with additional control over the aggregations.
        The dataset definition logic is left untouched by this method (by default: ["id", "domain", "processing_level", "xrfreq"]),
        except that newly aggregated columns are removed from the "id".
        This will override any "custom" id, ones not unstackable with :py:func:`~xscen.catalog.unstack_id`.

        Ensemble preprocessing logic is taken from :py:func:`xclim.ensembles.create_ensemble`.
        When `create_ensemble_on` is given, the function ensures all entries have the correct time coordinate according to `xrfreq`.

        Parameters
        ----------
        concat_on : list of str or str, optional
          A list of catalog columns over which to concat the datasets (in addition to 'time').
          Each will become a new dimension with the column values as coordinates.
          Xarray concatenation rules apply and can be acted upon through `xarray_combine_by_coords_kwargs`.
        create_ensemble_on : list of str or str, optional
          The given column values will be merged into a new id-like "realization" column, which will be concatenated over.
          The given columns are removed from the dataset id, to remove them from the groupby_attrs logic.
          Xarray concatenation rules apply and can be acted upon through `xarray_combine_by_coords_kwargs`.
        ensemble_name : list of strings, optional
          If `create_ensemble_on` is given, this can be a subset of those column names to use when constructing the realization coordinate.
          If None, this will be the same as `create_ensemble_on`.
          The resulting coordinate must be unique.
        calendar : str, optional
          If `create_ensemble_on` is given but not `preprocess`, all datasets are converted to this calendar before concatenation.
          Ignored otherwise (default). If None, no conversion is done. `align_on` is always "date".
          If `preprocess` is given, it must do the needed calendar handling.
        kwargs:
          Any other arguments are passed to :py:meth:`~intake_esm.core.esm_datastore.to_dataset_dict`.
          The `preprocess` argument must convert calendars as needed if `create_ensemble_on` is given.

        Returns
        -------
        :py:class:`~xarray.Dataset`

        See Also
        --------
        intake_esm.core.esm_datastore.to_dataset_dict
        intake_esm.core.esm_datastore.to_dask
        xclim.ensembles.create_ensemble
        """
        cat = deepcopy(self)
        # Put back what was removed by the copy...
        cat._requested_variables = self._requested_variables

        if isinstance(concat_on, str):
            concat_on = [concat_on]
        if isinstance(create_ensemble_on, str):
            create_ensemble_on = [create_ensemble_on]
        if ensemble_name is None:
            ensemble_name = create_ensemble_on
        elif not set(ensemble_name).issubset(create_ensemble_on):
            raise ValueError("`ensemble_name` must be a subset of `create_ensemble_on`.")
        rm_from_id = (concat_on or []) + (create_ensemble_on or []) + ["realization"]

        aggs = {agg.attribute_name for agg in cat.esmcat.aggregation_control.aggregations}
        if not set(cat.esmcat.aggregation_control.groupby_attrs).isdisjoint(rm_from_id):
            raise ValueError(f"Can't add aggregations for columns in the catalog's groupby_attrs ({cat.esmcat.aggregation_control.groupby_attrs})")
        if not aggs.isdisjoint(rm_from_id):
            raise ValueError(f"Can't add aggregations for columns were an aggregation already exists ({aggs})")

        if concat_on:
            cat.esmcat.aggregation_control.aggregations.extend(
                [intake_esm.cat.Aggregation(type=intake_esm.cat.AggregationType.join_new, attribute_name=col) for col in concat_on]
            )

        if create_ensemble_on:
            cat.df["realization"] = generate_id(cat.df, ensemble_name)
            cat.esmcat.aggregation_control.aggregations.append(
                intake_esm.cat.Aggregation(
                    type=intake_esm.cat.AggregationType.join_new,
                    attribute_name="realization",
                )
            )
            xrfreq = cat.df["xrfreq"].unique()[0]

            if kwargs.get("preprocess") is None:

                def preprocess(ds):
                    ds = ensure_correct_time(ds, xrfreq)
                    if calendar is not None:
                        ds = ds.convert_calendar(
                            calendar,
                            use_cftime=(calendar != "default"),
                            align_on="date",
                        )
                    return ds

                kwargs["preprocess"] = preprocess

        if len(rm_from_id) > 1:
            # Guess what the ID was and rebuild a new one, omitting the columns part of the aggregation
            unstacked = unstack_id(cat)
            cat.esmcat.df["id"] = cat.df.apply(
                lambda row: _build_id(row, [col for col in unstacked[row["id"]] if col not in rm_from_id]),
                axis=1,
            )

        if (N := len(cat.keys())) != 1:
            raise ValueError(f"Expected exactly one dataset, received {N} instead : {cat.keys()}")

        kwargs = _xarray_defaults(**kwargs)

        ds = cat.to_dask(**kwargs)
        return ds

    def copy_files(
        self,
        dest: str | os.PathLike,
        flat: bool = True,
        unzip: bool = False,
        zipzarr: bool = False,
        inplace: bool = False,
    ):
        """
        Copy each file of the catalog to another location, unzipping datasets along the way if requested.

        Parameters
        ----------
        cat: DataCatalog or ProjectCatalog
            A catalog to copy.
        dest: str, path
            The root directory of the destination.
        flat: bool
            If True (default), all dataset files are copied in the same directory.
            Renaming with an integer suffix ("{name}_01.{ext}") is done in case of duplicate file names.
            If False, :py:func:`xscen.catutils.build_path` (with default arguments) is used to generated the new path below the destination.
            Nothing is done in case of duplicates in that case.
        unzip: bool
            If True, any datasets with a `.zip` suffix are unzipped during the copy (or rather instead of a copy).
        zipzarr: bool
            If True, any datasets with a `.zarr` suffix are zipped during the copy (or rather instead of a copy).
        inplace : bool
            If True, the catalog is updated in place. If False (default), a copy is returned.

        Returns
        -------
        If inplace is False, this returns a catalog similar to self except with updated filenames. Some special attributes are not preserved,
        such as those added by :py:func:`xscen.extract.search_data_catalogs`. In this case, use `inplace=True`.
        """
        # Local imports to avoid circular imports
        from .catutils import build_path
        from .io import unzip_directory, zip_directory

        dest = Path(dest)
        data = self.esmcat._df.copy()
        if flat:
            new_paths = []
            for path in map(Path, data.path.values):
                if unzip and path.suffix == ".zip":
                    new = dest / path.with_suffix("").name
                elif zipzarr and path.suffix == ".zarr":
                    new = dest / path.with_suffix(".zarr.zip").name
                else:
                    new = dest / path.name
                if new in new_paths:
                    suffixes = "".join(new.suffixes)
                    name = new.name.removesuffix(suffixes)
                    i = 1
                    while new in new_paths:
                        new = dest / (name + f"_{i:02d}" + suffixes)
                        i += 1
                new_paths.append(new)
            data["new_path"] = new_paths
        else:
            data = build_path(data, root=dest).drop(columns=["new_path_type"])

        msg = f"Will copy {len(data)} files."
        logger.debug(msg)
        for _, row in data.iterrows():
            old = Path(row.path)
            new = Path(row.new_path)
            if unzip and old.suffix == ".zip":
                msg = f"Unzipping {old} to {new}."
                logger.info(msg)
                unzip_directory(old, new)
            elif zipzarr and old.suffix == ".zarr":
                msg = f"Zipping {old} to {new}."
                logger.info(msg)
                zip_directory(old, new)
            elif old.is_dir():
                msg = f"Copying directory tree {old} to {new}."
                logger.info(msg)
                sh.copytree(old, new)
            else:
                msg = f"Copying file {old} to {new}."
                logger.info(msg)
                sh.copy(old, new)
        if inplace:
            self.esmcat._df["path"] = data["new_path"]
            return
        data["path"] = data["new_path"]
        data = data.drop(columns=["new_path"])
        return self.__class__({"esmcat": self.esmcat.model_dump(), "df": data})


class ProjectCatalog(DataCatalog):
    """
    A DataCatalog with additional 'write' functionalities that can update and upload itself.

    See Also
    --------
    intake_esm.core.esm_datastore
    """

    @classmethod
    def create(
        cls,
        filename: os.PathLike | str,
        *,
        project: dict | None = None,
        overwrite: bool = False,
    ):
        r"""
        Create a new project catalog from some project metadata.

        Creates the json from default :py:data:`esm_col_data` and an empty csv file.

        Parameters
        ----------
        filename : os.PathLike or str
            A path to the json file (with or without suffix).
        project : dict, optional
            Metadata to create the catalog. If None, `CONFIG['project']` will be used.
            Valid fields are:

            - title : Name of the project, given as the catalog's "title".
            - id : slug-like version of the name, given as the catalog's id (should be url-proof)
              Defaults to a modified name.
            - version : Version of the project (and thus the catalog), string like "x.y.z".
            - description : Detailed description of the project, given to the catalog's "description".
            - Any other entry defined in :py:data:`esm_col_data`.

            At least one of `id` and `title` must be given, the rest is optional.
        overwrite : bool
            If True, will overwrite any existing JSON and CSV file.

        Returns
        -------
        ProjectCatalog
            An empty intake_esm catalog.
        """
        path = Path(filename)
        meta_path = path.with_suffix(".json")
        data_path = path.with_suffix(".csv")

        if (meta_path.is_file() or data_path.is_file()) and not overwrite:
            raise FileExistsError("Catalog file already exist (at least one of {meta_path} or {data_path}).")

        meta_path.parent.mkdir(parents=True, exist_ok=True)

        project = project or CONFIG.get("project") or {}

        if "id" not in project and "title" not in project:
            raise ValueError('At least one of "id" or "title" must be given in the metadata.')

        project["catalog_file"] = str(data_path)
        if "id" not in project:
            project["id"] = project.get("title", "").replace(" ", "")

        esmdata = recursive_update(esm_col_data.copy(), project)

        df = pd.DataFrame(columns=COLUMNS)

        cat = cls({"esmcat": esmdata, "df": df})  # TODO: Currently, this drops "version" because it is not a recognized attribute
        cat.serialize(
            path.stem,
            directory=path.parent,
            catalog_type="file",
            to_csv_kwargs={"compression": None},
        )

        # Change catalog_file to a relative path
        with Path(meta_path).open(encoding="utf-8") as f:
            meta = json.load(f)
            meta["catalog_file"] = data_path.name
        with Path(meta_path).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return cls(str(meta_path))

    def __init__(
        self,
        df: str | dict,
        *args,
        create: bool = False,
        overwrite: bool = False,
        project: dict | None = None,
        check_valid: bool = True,
        drop_duplicates: bool = True,
        **kwargs,
    ):
        r"""
        Open or create a project catalog.

        Parameters
        ----------
        df : str or dict
            If str, this must be a path or URL to a catalog JSON file.
            If dict, this must be a dict representation of an ESM catalog.  See the notes below.
        create : bool
            If True, and if 'df' is a string, this will create an empty ProjectCatalog if none already exists.
        overwrite : bool
            If this and 'create' are True, this will overwrite any existing JSON and CSV file with an empty catalog.
        project : dict, optional
            Metadata to create the catalog, if required.
        check_valid : bool
            If True (default), will check that all files in the catalog exist on disk and remove those that don't.
        drop_duplicates : bool
            If True (default), will drop duplicates in the catalog based on the 'id' and 'path' columns.
        \**kwargs : dict
            Any other arguments are passed to xscen.catalog.DataCatalog.


        Notes
        -----
        New ProjectCatalog must first be created empty, using 'df' as the path to the future JSON file, then populated using .update().
        The dictionary in 'df' must have two keys: ‘esmcat’ and ‘df’.
        The ‘esmcat’ key must be a dict representation of the ESM catalog. This should follow the template used by xscen.catalog.esm_col_data.
        The ‘df’ key must be a Pandas DataFrame containing content that would otherwise be in the CSV file.
        """
        if create:
            if isinstance(df, str | Path) and (not Path(df).is_file() or overwrite):
                self.create(df, project=project, overwrite=overwrite)
        super().__init__(
            df,
            *args,
            check_valid=check_valid,
            drop_duplicates=drop_duplicates,
            **kwargs,
        )
        self.check_valid_flag = check_valid
        self.drop_duplicates_flag = drop_duplicates
        self.meta_file = df if not isinstance(df, dict) else None

    # TODO: Implement a way to easily destroy part of the catalog to "reset" some steps
    def update(
        self,
        df: None | (DataCatalog | intake_esm.esm_datastore | pd.DataFrame | pd.Series | Sequence[pd.Series]) = None,
    ):
        """
        Update the catalog with new data and writes the new data to the csv file.

        Once the internal dataframe is updated with `df`, the csv on disk is parsed,
        updated with the internal dataframe, duplicates are dropped and everything is
        written back to the csv. This means that nothing is _removed_* from the csv when
        calling this method, and it is safe to use even with a subset of the catalog.

        Warnings
        --------
        If a file was deleted between the parsing of the catalog and this call,
        it will be removed from the csv if `check_valid` is called.

        Parameters
        ----------
        df : DataCatalog | intake_esm.esm_datastore | pd.DataFrame | pd.Series  | Sequence[pd.Series], optional
            Data to be added to the catalog. If None, nothing is added, but the catalog is still updated.
        """
        # Append the new DataFrame or Series
        if isinstance(df, DataCatalog) or isinstance(df, intake_esm.esm_datastore):
            self.esmcat._df = pd.concat([self.df, df.df])
        elif df is not None:
            if isinstance(df, pd.Series):
                df = pd.DataFrame(df).transpose()
            self.esmcat._df = pd.concat([self.df, df])
        if self.check_valid_flag:
            self.check_valid()
        if self.drop_duplicates_flag:
            self.drop_duplicates()

        # make sure year really has 4 digits
        if "date_start" in self.df:
            if os.name == "nt":
                y_format = "%Y"
            else:
                y_format = "%4Y"

            df_fix_date = self.df.copy()
            df_fix_date["date_start"] = pd.Series(
                [(x if isinstance(x, str) else "" if pd.isnull(x) else x.strftime(f"{y_format}-%m-%d %H:00")) for x in self.df.date_start]
            )

            df_fix_date["date_end"] = pd.Series(
                [(x if isinstance(x, str) else "" if pd.isnull(x) else x.strftime(f"{y_format}-%m-%d %H:00")) for x in self.df.date_end]
            )

            df_str = df_fix_date
        else:
            df_str = self.df

        if self.meta_file is not None:
            with fs.open(self.esmcat.catalog_file, "wb") as csv_outfile:
                df_str.to_csv(csv_outfile, index=False, compression=None)
        else:
            read_csv_kwargs = deepcopy(self.read_csv_kwargs)
            del read_csv_kwargs["parse_dates"]
            del read_csv_kwargs["date_parser"]
            # Update the catalog file saved on disk
            disk_cat = DataCatalog(
                {
                    "esmcat": self.esmcat.dict(),
                    "df": pd.read_csv(self.esmcat.catalog_file, **read_csv_kwargs),
                }
            )
            disk_cat.esmcat._df = pd.concat([disk_cat.df, df_str])
            if self.check_valid_flag:
                disk_cat.check_valid()
            if self.drop_duplicates_flag:
                disk_cat.drop_duplicates()
            with fs.open(disk_cat.esmcat.catalog_file, "wb") as csv_outfile:
                disk_cat.df.to_csv(csv_outfile, index=False, compression=None)

    def update_from_ds(
        self,
        ds: xr.Dataset,
        path: os.PathLike | str,
        info_dict: dict | None = None,
        **info_kwargs,
    ):
        """
        Update the catalog with new data and writes the new data to the csv file.

        We get the new data from the attributes of `ds`, the dictionary `info_dict` and `path`.

        Once the internal dataframe is updated with the new data, the csv on disk is parsed,
        updated with the internal dataframe, duplicates are dropped and everything is
        written back to the csv. This means that nothing is _removed_* from the csv when
        calling this method, and it is safe to use even with a subset of the catalog.

        Warnings
        --------
        If a file was deleted between the parsing of the catalog and this call,
        it will be removed from the csv if `check_valid` is called.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset that we want to add to the catalog.
            The columns of the catalog will be filled from the global attributes starting with 'cat:' of the dataset.
        info_dict : dict, optional
            Extra information to fill in the catalog.
        path : os.PathLike or str
            Path to the file that contains the dataset. This will be added to the 'path' column of the catalog.
        """
        d = {}

        for col in self.df.columns:
            if f"cat:{col}" in ds.attrs:
                d[col] = ds.attrs[f"cat:{col}"]
        if info_dict:
            d.update(info_dict)
        if info_kwargs:
            d.update(info_kwargs)

        if "time" in ds.dims:
            d["date_start"] = str(ds.isel(time=0).time.dt.strftime("%4Y-%m-%d %H:%M:%S").values)
            d["date_end"] = str(ds.isel(time=-1).time.dt.strftime("%4Y-%m-%d %H:%M:%S").values)

        d["path"] = str(Path(path))

        # variable should be based on the Dataset
        d["variable"] = tuple(v for v in ds.data_vars if len(ds[v].dims) > 0)

        if "format" not in d:
            d["format"] = Path(d["path"]).suffix.split(".")[1]
            msg = f"File format not specified. Adding it as '{d['format']}' based on file name."
            logger.info(msg)

        self.update(pd.Series(d))

    def refresh(self):
        """Reread the catalog CSV saved on disk."""
        if self.meta_file is None:
            raise ValueError("Only full catalogs can be refreshed, but this instance is only a subset.")
        self.esmcat = ESMCatalogModel.load(self.meta_file, read_csv_kwargs=self.read_csv_kwargs)
        initlen = len(self.esmcat.df)
        if self.check_valid_flag:
            self.check_valid()
        if self.drop_duplicates_flag:
            self.drop_duplicates()
        if len(self.df) != initlen:
            self.update()

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"<{self.esmcat.id or ''} project catalog with {len(self)} dataset(s) from "
            f"{len(self.df)} asset(s) ({'subset' if self.meta_file is None else 'full'})>"
        )


def concat_data_catalogs(*dcs):
    """
    Concatenate a multiple DataCatalogs.

    Output catalog is the union of all rows and all derived variables, with the "esmcat"
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
        requested_variables_true.extend(getattr(dc, "_requested_variables_true", []))
        dependent_variables.extend(getattr(dc, "_dependent_variables", []))
        requested_variable_freqs.extend(getattr(dc, "_requested_variable_freqs", []))
    df = pd.concat(catalogs, axis=0).drop_duplicates(ignore_index=True)
    dvr = intake_esm.DerivedVariableRegistry()
    dvr._registry.update(registry)
    newcat = DataCatalog({"esmcat": dcs[0].esmcat.model_dump(), "df": df}, registry=dvr)
    newcat._requested_variables = requested_variables
    if requested_variables_true:
        newcat._requested_variables_true = requested_variables_true
    if dependent_variables:
        newcat._dependent_variables = dependent_variables
    if requested_variable_freqs:
        newcat._requested_variable_freqs = requested_variable_freqs
    return newcat


def _build_id(element: pd.Series, columns: list[str]):
    """Build an ID from a catalog's row and a list of columns."""
    return "_".join(map(str, filter(pd.notna, element[columns].values)))


def generate_id(df: pd.DataFrame | xr.Dataset, id_columns: list | None = None) -> pd.Series:
    """
    Create an ID from column entries.

    Parameters
    ----------
    df: pd.DataFrame, xr.Dataset
      Data for which to create an ID.
    id_columns : list, optional
      List of column names on which to base the dataset definition. Empty columns will be skipped.
      If None (default), uses :py:data:`ID_COLUMNS`.

    Returns
    -------
    pd.Series
        A series of IDs, one per row of the input DataFrame.
    """
    if isinstance(df, xr.Dataset):
        df = pd.DataFrame.from_dict({key[4:]: [value] for key, value in df.attrs.items() if key.startswith("cat:")})

    id_columns = [x for x in (id_columns or ID_COLUMNS) if x in df.columns]

    return df.apply(_build_id, axis=1, args=(id_columns,))


def unstack_id(df: pd.DataFrame | ProjectCatalog | DataCatalog) -> dict:
    """
    Reverse-engineer an ID using catalog entries.

    Parameters
    ----------
    df : pd.DataFrame or ProjectCatalog or DataCatalog
        Either a Project/DataCatalog or a pandas DataFrame.

    Returns
    -------
    dict
        Dictionary with one entry per unique ID, which are themselves dictionaries of all the individual parts of the ID.
    """
    if isinstance(df, ProjectCatalog | DataCatalog):
        df = df.df

    out = {}
    for ids in pd.unique(df["id"]):
        subset = df[df["id"] == ids]

        # Only keep relevant columns
        subset = subset[[col for col in subset.columns if bool(re.search(f"((_)|(^)){subset[col].iloc[0]!s}((_)|($))", ids))]].drop("id", axis=1)

        # Make sure that all elements are the same, if there are multiple lines
        if not (subset.nunique() == 1).all():
            raise ValueError("Not all elements of the columns are the same for a given ID!")

        out[ids] = {attr: subset[attr].iloc[0] for attr in subset.columns}

    return out


def subset_file_coverage(
    df: pd.DataFrame,
    periods: list[str] | list[list[str]],
    *,
    coverage: float = 0.99,
    duplicates_ok: bool = False,
) -> pd.DataFrame:
    """
    Return a subset of files that overlap with the target periods.

    Parameters
    ----------
    df : pd.DataFrame
        List of files to be evaluated, with at least a date_start and date_end column,
        which are expected to be `datetime64` objects.
    periods : list of str or list of lists of str
        Either [start, end] or list of [start, end] for the periods to be evaluated.
        All periods must be covered, otherwise an empty subset is returned.
    coverage : float
        Percentage of hours that need to be covered in a given period for the dataset to be valid. Use 0 to ignore this checkup.
        The coverage calculation is only valid if there are no overlapping periods in `df` (ensure with `duplicates_ok=False`).
    duplicates_ok : bool
        If True, no checkup is done on possible duplicates.

    Returns
    -------
    pd.DataFrame
        Subset of files that overlap the targeted periods.
    """
    periods = standardize_periods(periods, out_dtype="datetime")

    # Create an Interval for each file
    intervals = pd.IntervalIndex.from_arrays(
        df["date_start"].astype("<M8[ms]"),
        df["date_end"].astype("<M8[ms]"),
        closed="both",
    )

    # Check for duplicated Intervals
    if duplicates_ok is False and intervals.is_overlapping:
        msg = f"{df['id'].iloc[0] + ': ' if 'id' in df.columns else ''}Time periods are overlapping."
        logging.warning(msg)
        return pd.DataFrame(columns=df.columns)

    # Create an array of True/False
    files_to_keep = []
    for period in periods:
        period_interval = pd.Interval(
            date_parser(period[0]),
            date_parser(period[1], end_of_period=True),
            closed="both",
        )
        files_in_range = intervals.overlaps(period_interval)

        if not files_in_range.any():
            msg = f"{df['id'].iloc[0] + ': ' if 'id' in df.columns else ''}Insufficient coverage (no files in range {period})."
            logging.warning(msg)
            return pd.DataFrame(columns=df.columns)

        # Very rough guess of the coverage relative to the requested period,
        # without having to open the files or checking day-by-day
        if coverage > 0:
            # Number of hours in the requested period
            period_length = period_interval.length
            # Sum of hours in all selected files, restricted by the requested period
            guessed_length = pd.IntervalIndex.from_arrays(
                intervals[files_in_range].map(lambda x: max(x.left, period_interval.left)).astype("<M8[ms]"),  # noqa: B023 # FIXME
                intervals[files_in_range].map(lambda x: min(x.right, period_interval.right)).astype("<M8[ms]"),  # noqa: B023 # FIXME
            ).length.sum()

            if guessed_length / period_length < coverage:
                msg = (
                    f"{df['id'].iloc[0] + ': ' if 'id' in df.columns else ''}Insufficient coverage (guessed at {guessed_length / period_length:.1%})."
                )
                logging.warning(msg)
                return pd.DataFrame(columns=df.columns)

        files_to_keep.append(files_in_range)

    return df[reduce(or_, files_to_keep)]
