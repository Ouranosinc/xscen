import datetime
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Union

import clisops.core.subset
import numpy as np
import pandas as pd
import xarray as xr
import xclim as xc
from intake_esm.derived import DerivedVariableRegistry

from . import CV
from .catalog import (
    ID_COLUMNS,
    DataCatalog,
    concat_data_catalogs,
    date_parser,
    generate_id,
    parse_from_ds,
)
from .common import natural_sort
from .config import parse_config
from .indicators import load_xclim_module, registry_from_module

logger = logging.getLogger(__name__)


@parse_config
def search_data_catalogs(
    data_catalogs: Union[list, DataCatalog],
    variables_and_freqs: dict,
    *,
    other_search_criteria: Optional[dict] = None,
    exclusions: dict = None,
    match_hist_and_fut: bool = False,
    periods: list = None,
    id_columns: Optional[List[str]] = None,
    allow_resampling: bool = True,
    allow_conversion: bool = True,
    restrict_resolution: str = None,
    restrict_members: dict = None,
) -> dict:
    """
    Search through DataCatalogs.

    Parameters
    ----------
    data_catalogs : Union[list, DataCatalog]
      DataCatalog (or multiple, in a list) or path to JSON data catalogs. They must use the same columns.
    variables_and_freqs : dict
      Variables and freqs to search for, following a 'variable: xr-freq-compatible-str' format.
    other_search_criteria : dict, optional
      Other criteria to search for in the catalogs' columns, following a 'column_name: list(subset)' format.
    exclusions : dict, optional
      Same as other_search_criteria, but for eliminating results.
    match_hist_and_fut: bool, optional
      If True, historical and future simulations will be combined into the same line, and search results lacking one of them will be rejected.
    periods : list
      [start, end] of the period to be evaluated (or a list of lists)
    id_columns : list, optional
      List of columns used to create a id column. If None is given, the original
      "id" is left.
    allow_resampling: bool
      If True (default), variables with a higher time resolution than requested are considered.
    allow_conversion: bool
      If True (default) and if the requested variable cannot be found, intermediate variables are
      searched given that there exists a converting function in the "derived variable registry"
      defined by "xclim_modules/conversions.yml".
    restrict_resolution: str
      Used to restrict the results to the finest/coarsest resolution available for a given simulation.
      ['finest', 'coarsest'].
    restrict_members: dict
      Used to restrict the results to a given number of members for a given simulation.
      Currently only supports {"ordered": int} format.

    Note
    ----------
    - The "other_search_criteria" argument accepts wildcard (*) and regular expressions.
    - Frequency can be wildcarded with 'NA' in the `variables_and_freqs` dict.
    - Variable names cannot be wildcarded, they must be CMIP6-standard.

    Returns
    -------
    dict
        Keys are the id and values are the DataCatalogs for each entry.
        A single DataCatalog can be retrieved with `concat_data_catalogs(*out.values())`.
        Each DataCatalog has a subset of the derived variable registry that corresponds
        to the needs of this specific group.
        Usually, each entry can be written to file in a single Dataset when using
        `extract_dataset` with the same arguments.
    """
    cat_kwargs = {}
    if allow_conversion:
        cat_kwargs = {
            "registry": registry_from_module(
                load_xclim_module(
                    Path(__file__).parent / "xclim_modules" / "conversions"
                )
            )
        }

    # Prepare a unique catalog to search from, with the DerivedCat added if required
    if isinstance(data_catalogs, DataCatalog):
        paths = [data_catalogs.esmcat.catalog_file]
    elif isinstance(data_catalogs, list) and all(
        isinstance(dc, DataCatalog) for dc in data_catalogs
    ):
        paths = [dc.esmcat.catalog_file for dc in data_catalogs]
    elif isinstance(data_catalogs, list) and all(
        isinstance(dc, str) for dc in data_catalogs
    ):
        paths = [DataCatalog(dc).esmcat.catalog_file for dc in data_catalogs]
    else:
        raise ValueError("Catalogs type not recognized.")
    catalog = DataCatalog.from_csv(paths, name="source", **cat_kwargs)
    logger.info(f"Catalog opened: {catalog} from {len(paths)} files.")

    if match_hist_and_fut:
        logger.info("Dispatching historical dataset to future experiments.")
        catalog = dispatch_historical_to_future(catalog, id_columns)

    # Cut entries that do not match search criteria
    if other_search_criteria:
        catalog = catalog.search(**other_search_criteria)
        logger.info(
            f"{len(catalog.df)} assets matched the criteria : {other_search_criteria}."
        )
    if exclusions:
        ex = catalog.search(**exclusions)
        catalog.esmcat._df = pd.concat([catalog.df, ex.df]).drop_duplicates(keep=False)
        logger.info(
            f"Removing {len(ex.df)} assets based on exclusion dict : {exclusions}."
        )

    ids = generate_id(catalog.df, id_columns)
    if id_columns is not None:
        # Recreate id from user specifications
        catalog.df["id"] = ids
    else:
        # Only fill in the missing IDs
        catalog.df["id"] = catalog.df["id"].fillna(ids)

    logger.info(f"Iterating over {catalog.nunique()['id']} potential datasets.")
    # Loop on each dataset to assess whether they have all required variables
    # And select best freq/timedelta for each
    catalogs = {}
    for (sim_id,), scat in catalog.iter_unique("id"):
        # Find all the entries that match search parameters
        varcats = []
        for var_id, xrfreq in variables_and_freqs.items():
            if xrfreq == "fx":
                varcat = scat.search(
                    xrfreq=xrfreq,
                    variable=var_id,
                    require_all_on=["id", "xrfreq"],
                )
                # TODO: Temporary fix until this is changed in intake_esm
                varcat._requested_variables_true = [var_id]
                varcat._dependent_variables = list(
                    set(varcat._requested_variables).difference(
                        varcat._requested_variables_true
                    )
                )
            else:
                # TODO: Add support for DerivedVariables that themselves require DerivedVariables
                # TODO: Add support for DerivedVariables that exist on different frequencies (e.g. 1hr 'pr' & 3hr 'tas')
                varcat = scat.search(variable=var_id, require_all_on=["id", "xrfreq"])
                logger.debug(
                    f"At var {var_id}, after search cat has {varcat.derivedcat.keys()}"
                )
                # TODO: Temporary fix until this is changed in intake_esm
                varcat._requested_variables_true = [var_id]
                varcat._dependent_variables = list(
                    set(varcat._requested_variables).difference(
                        varcat._requested_variables_true
                    )
                )

                # We want to match lines with the correct freq,
                # IF allow_resampling is True and xrfreq translates to a timedelta,
                # we also want those with (stricyly) higher temporal resolution
                same_frq = varcat.df.xrfreq == xrfreq
                td = pd.to_timedelta(CV.xrfreq_to_timedelta(xrfreq))
                varcat.df["timedelta"] = pd.to_timedelta(
                    varcat.df.xrfreq.apply(CV.xrfreq_to_timedelta, default="NAN")
                )
                # else is joker (any timedelta)
                lower_frq = (
                    np.less(varcat.df.timedelta, td) if pd.notnull(td) else False
                )
                varcat.esmcat._df = varcat.df[same_frq | (lower_frq & allow_resampling)]

                # For each dataset (id - xrfreq - processing_level - domain - variable), make sure that file availability covers the requested time periods
                if periods is not None and len(varcat) > 0:
                    valid_tp = []
                    for var, group in varcat.df.groupby(
                        varcat.esmcat.aggregation_control.groupby_attrs + ["variable"]
                    ):
                        valid_tp.append(
                            _subset_file_coverage(group, periods)
                        )  # If valid, this returns the subset of files that cover the time period
                    varcat.esmcat._df = pd.concat(valid_tp)

                # We now select the coarsest timedelta for each variable
                # we need to re-iterate over variables in case we used the registry (and thus there are multiple variables in varcat)
                rows = []
                for var, group in varcat.df.groupby("variable"):
                    rows.append(group[group.timedelta == group.timedelta.max()])
                if rows:
                    # check if the requested variable exists and if so, remove DeriveVariable references
                    v_list = [rows[i]["variable"].iloc[0] for i in range(len(rows))]
                    v_list_check = [
                        var_id in v_list[i] for i in range(len(v_list))
                    ]  # necessary in case a file has multiple variables
                    if any(v_list_check):
                        rows = [rows[v_list_check.index(True)]]
                        varcat.derivedcat = DerivedVariableRegistry()
                    varcat.esmcat._df = pd.concat(rows, ignore_index=True)
                else:
                    varcat.esmcat._df = pd.DataFrame()

            if varcat.df.empty:
                logger.info(
                    f"Dataset {sim_id} doesn't have all needed variables (missing at least {var_id})."
                )
                break
            if "timedelta" in varcat.df.columns:
                varcat.df.drop(columns=["timedelta"], inplace=True)
            varcat._requested_variable_freqs = [xrfreq]
            varcats.append(varcat)
        else:
            catalogs[sim_id] = concat_data_catalogs(*varcats)
            if periods is not None:
                if not isinstance(periods[0], list):
                    periods = [periods]
                catalogs[sim_id]._requested_periods = periods

    logger.info(
        f"Found {len(catalogs)} with all variables requested and corresponding to the criteria."
    )

    if restrict_resolution is not None:
        catalogs = restrict_by_resolution(catalogs, id_columns, restrict_resolution)

    if restrict_members is not None:
        catalogs = restrict_multimembers(catalogs, id_columns, restrict_members)

    return catalogs


def dispatch_historical_to_future(catalog: DataCatalog, id_columns: list):
    """Updates a DataCatalog by recopying each "historical" entry to its corresponding future experiments.

    For examples, if an historical entry has corresonding "ssp245" and "ssp585" entries,
    then it is copied twice, with its "experiment" field modified accordingly.
    The original "historical" entry is removed. This way, a subsequent search of the catalog
    with "experiment='ssp245'" includes the _historical_ assets (with no apparent distinction).

    "Historical" assets that did not find a match are removed from the output catalog.
    """
    expcols = [
        "experiment",
        "id",
        "variable",
        "xrfreq",
        "date_start",
        "date_end",
        "path",
        "format",
        "activity",
    ]  # These columns can differentiate the historical from the future.
    sim_id_no_exp = list(
        filter(
            lambda c: c not in expcols,
            set(id_columns or ID_COLUMNS).intersection(catalog.df.columns),
        )
    )
    # For each unique element from this column list, the historical shouldn't be distinct
    # from the future scenario, allowing us to match them.
    # "Same hist member" as in "each future realization stems from the same historical member"

    df = catalog.df.copy()
    df["same_hist_member"] = df[sim_id_no_exp].apply(
        lambda row: "_".join(row.values.astype(str)), axis=1
    )

    new_lines = []
    for group in df.same_hist_member.unique():
        sdf = df[df.same_hist_member == group]
        hist = sdf[sdf.experiment == "historical"]
        if hist.empty:
            continue
        for activity_id in set(sdf.activity) - {"HighResMip", np.NaN}:
            sub_sdf = sdf[sdf.activity == activity_id]
            for exp_id in set(sub_sdf.experiment) - {"historical", "piControl", np.NaN}:

                exp_hist = hist.copy()
                exp_hist["experiment"] = exp_id
                exp_hist["activity"] = activity_id
                exp_hist["same_hist_member"] = group
                sim_ids = sub_sdf[sub_sdf.experiment == exp_id].id.unique()
                if len(sim_ids) > 1:
                    raise ValueError(
                        f"Got multiple dataset ids where we expected only one... : {sim_ids}"
                    )
                exp_hist["id"] = sim_ids[0]
                new_lines.append(exp_hist)

    df = pd.concat([df] + new_lines, ignore_index=True).drop(
        columns=["same_hist_member"]
    )
    df = df[df.experiment != "historical"].reset_index(drop=True)
    return DataCatalog(
        {"esmcat": catalog.esmcat.dict(), "df": df},
        registry=catalog.derivedcat,
        drop_duplicates=False,
    )


def restrict_by_resolution(catalogs: dict, id_columns: list, restrictions: str):
    """Updates the results from search_data_catalogs by removing simulations with multiple resolutions available.

    Notes
    -----
    Currently supports:
        CMIP5 and CMIP6 (in that order, giving the highest priority to elements on the left):
            [gn, gn{a/g}, gr, gr{0-9}{a/g}, global, gnz, gr{0-9}z, gm]
        CORDEX:
            [DOM-{resolution}, DOM-{resolution}i]
    """
    df = pd.concat([catalogs[s].df for s in catalogs.keys()])
    # remove the domain from the group_by
    df["id_nodom"] = df[
        list(
            set(id_columns or ID_COLUMNS)
            .intersection(df.columns)
            .difference(["domain"])
        )
    ].apply(lambda row: "_".join(map(str, filter(pd.notna, row.values))), axis=1)
    for i in pd.unique(df["id_nodom"]):
        df_sim = df[df["id_nodom"] == i]
        domains = pd.unique(df_sim["domain"])

        if len(domains) > 1:
            logger.info(f"Dataset {i} appears to have multiple resolutions.")

            # For CMIP, the order is dictated by a list of grid labels
            if pd.unique(df_sim["activity"])[0] == "CMIP":
                order = np.array([])
                for d in domains:
                    match = [
                        CV.infer_resolution("CMIP").index(r)
                        for r in CV.infer_resolution("CMIP")
                        if re.match(pattern=r, string=d)
                    ]
                    if len(match) != 1:
                        raise ValueError(f"'{d}' matches no known CMIP domain.")
                    order = np.append(order, match[0])

                if restrictions == "finest":
                    chosen = [
                        np.sort(
                            np.array(domains)[
                                np.where(order == np.array(order).min())[0]
                            ]
                        )[0]
                    ]
                elif restrictions == "coarsest":
                    chosen = [
                        np.sort(
                            np.array(domains)[
                                np.where(order == np.array(order).max())[0]
                            ]
                        )[-1]
                    ]
                else:
                    raise ValueError(
                        "'restrict_resolution' should be 'finest' or 'coarsest'"
                    )

            # For CORDEX, the order is dictated by both the grid label and the resolution itself (as well as the domain name)
            elif pd.unique(df_sim["activity"])[0] == "CORDEX":
                # Unique CORDEX domains
                cordex_doms = pd.unique([d.split("-")[0] for d in domains])
                chosen = []

                for d in cordex_doms:
                    sub = [doms for doms in domains if d in doms]
                    order = [
                        int(re.split("^([A-Z]{3})-([0-9]{2})([i]{0,1})$", s)[2])
                        for s in sub
                    ]

                    if restrictions == "finest":
                        chosen.extend(
                            [
                                np.sort(
                                    np.array(sub)[
                                        np.where(order == np.array(order).min())[0]
                                    ]
                                )[0]
                            ]
                        )
                    elif restrictions == "coarsest":
                        chosen.extend(
                            [
                                np.sort(
                                    np.array(sub)[
                                        np.where(order == np.array(order).max())[0]
                                    ]
                                )[0]
                            ]
                        )
                    else:
                        raise ValueError(
                            "'restrict_resolution' should be 'finest' or 'coarsest'"
                        )

            else:
                logger.warning(
                    f"Dataset {i} seems to have multiple resolutions, but its activity is not recognized or supported yet."
                )
                chosen = list(domains)
                pass

            to_remove = pd.unique(
                df_sim[
                    df_sim["domain"].isin(
                        list(set(pd.unique(df_sim["domain"])).difference(chosen))
                    )
                ]["id"]
            )

            for k in to_remove:
                logger.info(f"Removing {k} from the results.")
                catalogs.pop(k)

    return catalogs


def restrict_multimembers(catalogs: dict, id_columns: list, restrictions: dict):
    """Updates the results from search_data_catalogs by removing simulations with multiple members available

    Uses regex to try and adequately detect and order the member's identification number, but only tested for 'r-i-p'.

    """

    df = pd.concat([catalogs[s].df for s in catalogs.keys()])
    # remove the member from the group_by
    df["id_nomem"] = df[
        list(
            set(id_columns or ID_COLUMNS)
            .intersection(df.columns)
            .difference(["member"])
        )
    ].apply(lambda row: "_".join(map(str, filter(pd.notna, row.values))), axis=1)

    for i in pd.unique(df["id_nomem"]):
        df_sim = df[df["id_nomem"] == i]
        members = pd.unique(df_sim["member"])

        if len(members) > 1:
            logger.info(
                f"Dataset {i} has {len(members)} valid members. Restricting as per requested."
            )

            if "ordered" in restrictions:
                members = natural_sort(members)[0 : restrictions["ordered"]]
            else:
                raise NotImplementedError(
                    "Subsetting multiple members currently only supports 'ordered'."
                )

            to_remove = pd.unique(
                df_sim[
                    df_sim["member"].isin(
                        list(set(pd.unique(df_sim["member"])).difference(members))
                    )
                ]["id"]
            )

            for k in to_remove:
                logger.info(f"Removing {k} from the results.")
                catalogs.pop(k)

    return catalogs


@parse_config
def extract_dataset(
    catalog: DataCatalog,
    *,
    variables_and_freqs: dict = None,
    periods: list = None,
    region: Optional[dict] = None,
    to_level: str = "extracted",
    xr_open_kwargs: dict = None,
    xr_combine_kwargs: dict = None,
    preprocess: Callable = None,
) -> Union[dict, xr.Dataset]:
    """
    Takes one element of the output of `search_data_catalogs` and returns a dataset,
    performing conversions and resampling as needed.

    Nothing is written to disk within this function.

    Parameters
    ----------
    catalog: DataCatalog
      Sub-catalog for a single dataset, one value of the output of `search_data_catalogs`.
    variables_and_freqs : dict
      Variables and freqs, following a 'variable: xrfreq-compatible str' format.
      If None, it will be read from catalog._requested_variables and catalog._requested_variable_freqs
      (set by `variables_and_freqs` in `search_data_catalogs`)
    periods : list
      [start, end] of the period to be evaluated (or a list of lists)
      Will be read from catalog._requested_periods if None. Leave both None to extract everything.
    region : dict, optional
      Description of the region and the subsetting method (required fields listed in the Notes).
    to_level: str
      The processing level to assign to the output.
      Defaults to 'extracted'
    xr_open_kwargs : dict, optional
      A dictionary of keyword arguments to pass to `DataCatalogs.to_dataset_dict`, which
      will be passed to `xr.open_dataset`.
    xr_combine_kwargs : dict, optional
      A dictionary of keyword arguments to pass to `DataCatalogs.to_dataset_dict`, which
      will be passed to `xr.combine_by_coords`.
    preprocess : callable, optional
      If provided, call this function on each dataset prior to aggregation.

    Returns
    -------
    dict, xr.Dataset
      Dictionary (keys = xrfreq) with datasets containing all available and computed variables,
      subsetted to the region, everything resampled to the requested frequency.
      If there is a single frequency, a Dataset will be returned instead.

    Notes
    -----
    'region' fields:
        name: str
            Region name used to overwrite domain in the catalog.
        method: str
            ['gridpoint', 'bbox', shape']
        <method>: dict
            Arguments specific to the method used.
        buffer: float, optional
            Multiplier to apply to the model resolution.
    """
    # Checks
    # must have all the same processing level and same id
    unique = catalog.unique()
    if len(unique["processing_level"]) > 1 or len(unique["id"]) > 1:
        raise ValueError(
            "The extraction catalog must have a unique id and unique processing_level."
        )
    if region is None and len(unique["domain"]) > 1:
        raise ValueError(
            "If a subset region is not given, the extraction catalog must have a unique domain."
        )

    if variables_and_freqs is None:
        try:
            variables_and_freqs = dict(
                zip(
                    catalog._requested_variables_true, catalog._requested_variable_freqs
                )
            )
        except ValueError:
            raise ValueError("Failed to determine the requested variables and freqs.")

    # Default arguments to send xarray
    xr_open_kwargs = xr_open_kwargs or {}
    xr_combine_kwargs = xr_combine_kwargs or {}
    xr_combine_kwargs.setdefault("data_vars", "minimal")

    # Open the catalog
    ds_dict = catalog.to_dataset_dict(
        xarray_open_kwargs=xr_open_kwargs,
        xarray_combine_by_coords_kwargs=xr_combine_kwargs,
        preprocess=preprocess,
    )

    out_dict = {}
    for xrfreq in pd.unique(list(variables_and_freqs.values())):
        ds = xr.Dataset()
        attrs = {}
        # iterate on the datasets, in reverse timedelta order
        for key, ds_ts in sorted(
            ds_dict.items(),
            key=lambda kv: CV.xrfreq_to_timedelta(
                catalog[kv[0]].df.xrfreq.iloc[0], default="NAN"
            ),
            reverse=True,
        ):
            for var_name, da in ds_ts.data_vars.items():
                # Support for grid_mapping, crs, and other such variables
                if len(da.dims) == 0 and var_name not in ds:
                    if bool(re.search("^|S[0-9]{1,4}$", str(da.dtype))):
                        da = da.astype("U")
                    ds = ds.assign({var_name: da})
                    continue

                # TODO: 2nd part is a temporary fix until this is changed in intake_esm
                if (
                    var_name in ds
                    or var_name not in catalog._requested_variables_true
                    or variables_and_freqs[var_name] != xrfreq
                ):
                    continue

                # TODO: This is a temporary fix for an xclim bug where the grid_mapping attribute is not transferred upon calculation
                if (
                    var_name in catalog.derivedcat
                    and len(
                        set(ds_ts.data_vars).intersection(
                            catalog.derivedcat[var_name].query["variable"]
                        )
                    )
                    >= 1
                ):
                    grid_mapping = np.unique(
                        [
                            ds_ts[v].attrs["grid_mapping"]
                            for v in set(ds_ts.data_vars).intersection(
                                catalog.derivedcat[var_name].query["variable"]
                            )
                            if "grid_mapping" in ds_ts[v].attrs
                        ]
                    )
                    if len(grid_mapping) == 1:
                        da.attrs["grid_mapping"] = grid_mapping[0]
                    elif len(grid_mapping) > 1:
                        raise ValueError("Multiple grid_mapping detected.")

                if "time" not in da.dims:
                    ds = ds.assign({var_name: da})
                else:  # check if it needs resampling
                    if pd.to_timedelta(
                        CV.xrfreq_to_timedelta(catalog[key].df["xrfreq"].iloc[0])
                    ) < pd.to_timedelta(
                        CV.xrfreq_to_timedelta(variables_and_freqs[var_name])
                    ):
                        logger.info(
                            f"Resampling {var_name} from [{catalog[key].df['xrfreq'].iloc[0]}]"
                            f" to [{variables_and_freqs[var_name]}]."
                        )
                        ds = ds.assign(
                            {
                                var_name: resample(
                                    da, variables_and_freqs[var_name], ds=ds_ts
                                )
                            }
                        )
                    elif (
                        catalog[key].df["xrfreq"].iloc[0]
                        == variables_and_freqs[var_name]
                    ):
                        ds = ds.assign({var_name: da})
                    else:
                        raise ValueError(
                            "Variable is at a coarser frequency than requested."
                        )

                attrs = ds_ts.attrs

        ds.attrs = attrs
        if "time" not in ds.dims:
            ds.attrs["cat/frequency"] = "fx"
            ds.attrs["cat/xrfreq"] = "fx"
        else:
            ds.attrs["cat/xrfreq"] = xrfreq
            ds.attrs["cat/frequency"] = CV.xrfreq_to_frequency(xrfreq)

        # Subset time on the periods
        if periods is None and hasattr(catalog, "_requested_periods"):
            periods = catalog._requested_periods
        if periods is not None and "time" in ds:
            if not isinstance(periods[0], list):
                periods = [periods]
            slices = []
            for period in periods:
                slices.extend([ds.sel({"time": slice(str(period[0]), str(period[1]))})])
            ds = xr.concat(slices, dim="time", **xr_combine_kwargs)

        # Custom call to clisops
        if region is not None:
            ds = clisops_subset(ds, region)
            ds.attrs["cat/domain"] = region["name"]

        # add relevant attrs
        ds.attrs["cat/processing_level"] = to_level
        if "cat/variable" not in ds.attrs:
            ds.attrs["cat/variable"] = parse_from_ds(ds, ["variable"])["variable"]

        out_dict[xrfreq] = ds

    return out_dict


@parse_config
def resample(
    da: xr.DataArray,
    target_frequency: str,
    *,
    ds: Optional[xr.Dataset] = None,
    method: Optional[str] = None,
) -> xr.DataArray:
    """
    Aggregate variable to the target frequency.

    Parameters
    ----------
    da: xr.DataArray
      DataArray of the variable to resample, must have a "time" dimension and be of a
      finer temporal resolution than "target_timestep".
    target_frequency: str
      The target frequency/freq str, must be one of the frequency supported by pandas.
    ds : xr.Dataset, optional
      The "wind_direction" resampling method needs extra variables, which can be given here.
    method : {'mean', 'min', 'max', 'sum', 'wind_direction'}, optional
      The resampling method. If None (default), it is guessed from the variable name,
      using the mapping in CVs/resampling_methods.json. If the variable is not found there,
      "mean" is used by default.

    Returns
    -------
    xr.DataArray
      Resampled variable

    """
    var_name = da.name

    if method is None:
        if var_name in CV.resampling_methods.dict:
            method = CV.resampling_methods(var_name)
            logger.info(
                f"Resampling method for {var_name}: '{method}', based on variable name."
            )
        else:
            method = "mean"
            logger.info(f"Resampling method for {var_name} defaulted to: 'mean'.")

    # TODO : Support non-surface wind?
    if method == "wind_direction":
        ds[var_name] = da
        if ds is None or not any(
            [
                all(v in ds for v in ["uas", "vas"]),
                all(v in ds for v in ["sfcWind", "sfcWindfromdir"]),
            ]
        ):
            raise ValueError(
                "Resampling method 'wind_direction' failed to find all required variables."
            )

        # The method requires uas, vas, and sfcWind. Acquire all of them.
        if all(v in ds for v in ["uas", "vas"]):
            uas, vas = ds.uas, ds.vas
        else:
            uas, vas = xc.indicators.atmos.wind_vector_from_speed(
                ds.sfcWind, ds.sfcWindfromdir
            )
        if "sfcWind" not in ds:
            ds["sfcWind"], _ = xc.indicators.atmos.wind_speed_from_vector(
                uas=ds["uas"], vas=ds["vas"]
            )

        # Resample first to find the average wind speed and components
        ds = ds.resample(time=target_frequency).mean(dim="time", keep_attrs=True)

        # Based on Vector Magnitude and Direction equations
        # For example: https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:vectors/x9e81a4f98389efdf:component-form/a/vector-magnitude-and-direction-review

        uas_attrs = ds["uas"].attrs
        vas_attrs = ds["vas"].attrs

        # Using the direction angle theta and the vector magnitude (sfcWind), re-adjust uas/vas
        theta = np.arctan2(ds["vas"], ds["uas"])
        ds["uas"] = ds["sfcWind"] * np.cos(theta)
        ds["uas"].attrs = uas_attrs
        ds["vas"] = ds["sfcWind"] * np.sin(theta)
        ds["vas"].attrs = vas_attrs

        # Prepare output
        if var_name in ["sfcWindfromdir"]:
            _, out = xc.indicators.atmos.wind_speed_from_vector(uas=uas, vas=vas)
        else:
            out = ds[var_name]

    else:
        out = getattr(da.resample(time=target_frequency), method)(
            dim="time", keep_attrs=True
        )

    initial_frequency = xr.infer_freq(da.time.dt.round("T")) or "undetected"

    new_history = (
        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {method} "
        f"resample from {initial_frequency} to {target_frequency} - xarray v{xr.__version__}"
    )
    history = (
        new_history + " \n " + out.attrs["history"]
        if "history" in out.attrs
        else new_history
    )
    out.attrs["history"] = history

    return out


def _subset_file_coverage(
    df: pd.DataFrame, periods: list, *, coverage: float = 0.99
) -> pd.DataFrame:
    """
    Returns a subset of files that overlap with the target period(s),
    The minimum resolution for periods is 1 hour.

    Parameters
    ----------
    df : pd.DataFrame
      List of files to be evaluated, with at least a date_start and date_end column,
      which are expected to be `pd.Period` objects with `freq='H'`.
    periods : list
      [start, end] of the period to be evaluated (or a list of lists)
    coverage : float
      Percentage of hours that need to be covered in a given period for the dataset to be valid

    Returns
    -------
    pd.DataFrame
      Subset of files that overlap the targetted period(s)
    """
    if not isinstance(periods[0], list):
        periods = [periods]

    # Create an Interval for each file

    file_intervals = df.apply(
        lambda r: pd.Interval(
            left=r["date_start"].ordinal, right=r["date_end"].ordinal, closed="both"
        ),
        axis=1,
    )

    # Check for duplicated Intervals
    if any(file_intervals.duplicated()):
        raise ValueError("Time periods are overlapping.")

    # Create an array of True/False
    files_to_keep = np.zeros(len(file_intervals), dtype=bool)
    for period in periods:
        period_interval = pd.Interval(
            left=date_parser(str(period[0]), freq="H").ordinal,
            right=date_parser(str(period[1]), end_of_period=True, freq="H").ordinal,
            closed="both",
        )
        files_in_range = file_intervals.apply(lambda r: period_interval.overlaps(r))

        # Very rough guess of the coverage relative to the requested period,
        # without having to open the files or checking day-by-day
        guessed_nb_hrs = np.min(
            [
                df[files_in_range]["date_end"].max(),
                date_parser(str(period[1]), end_of_period=True, freq="H"),
            ]
        ) - np.max(
            [
                df[files_in_range]["date_start"].min(),
                date_parser(str(period[0]), freq="H"),
            ]
        )
        period_nb_hrs = date_parser(
            str(period[1]), end_of_period=True, freq="H"
        ) - date_parser(str(period[0]), freq="H")

        # 'coverage' adds some leeway, for example to take different calendars into account or missing 2100-12-31
        if guessed_nb_hrs / period_nb_hrs < coverage or len(df[files_in_range]) == 0:
            return pd.DataFrame(columns=df.columns)

        files_to_keep = files_to_keep | files_in_range

    return df[files_to_keep]


def clisops_subset(ds: xr.Dataset, region: dict) -> xr.Dataset:
    """
    Custom call to clisops.subset() that allows for an automatic buffer around the region.

    Parameters
    ----------
    ds : xr.Dataset
      Dataset to be subsetted
    region : dict
      Description of the region and the subsetting method (required fields listed in the Notes)

    Note
    ----------
    'region' fields:
        method: str
            ['gridpoint', 'bbox', shape']
        <method>: dict
            Arguments specific to the method used.
        buffer: float, optional
            Multiplier to apply to the model resolution.

    Returns
    -------
    xr.Dataset
      Subsetted Dataset

    """
    if "buffer" in region.keys():
        # estimate the model resolution
        if len(ds.lon.dims) == 1:  # 1D lat-lon
            lon_res = np.abs(ds.lon.diff("lon")[0].values)
            lat_res = np.abs(ds.lat.diff("lat")[0].values)
        else:
            lon_res = np.abs(ds.lon[0, 0].values - ds.lon[0, 1].values)
            lat_res = np.abs(ds.lat[0, 0].values - ds.lat[1, 0].values)

    kwargs = deepcopy(region[region["method"]])

    if region["method"] in ["gridpoint"]:
        ds_subset = clisops.core.subset_gridpoint(ds, **kwargs)
        new_history = (
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{region['method']} spatial subsetting on {len(region['gridpoint']['lon'])} coordinates - clisops v{clisops.__version__}"
        )

    elif region["method"] in ["bbox"]:
        if "buffer" in region.keys():
            # adjust the boundaries
            kwargs["lon_bnds"] = (
                kwargs["lon_bnds"][0] - lon_res * region["buffer"],
                kwargs["lon_bnds"][1] + lon_res * region["buffer"],
            )
            kwargs["lat_bnds"] = (
                kwargs["lat_bnds"][0] - lat_res * region["buffer"],
                kwargs["lat_bnds"][1] + lat_res * region["buffer"],
            )

        ds_subset = clisops.core.subset_bbox(ds, **kwargs)
        new_history = (
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{region['method']} spatial subsetting with {'buffer=' + str(region['buffer']) if 'buffer' in region else 'no buffer'}"
            f", lon_bnds={np.array(region['bbox']['lon_bnds'])}, lat_bnds={np.array(region['bbox']['lat_bnds'])}"
            f" - clisops v{clisops.__version__}"
        )

    elif region["method"] in ["shape"]:
        if "buffer" in region.keys():
            kwargs["buffer"] = np.max([lon_res, lat_res]) * region["buffer"]

        ds_subset = clisops.core.subset_shape(ds, **kwargs)
        new_history = (
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{region['method']} spatial subsetting with {'buffer=' + str(region['buffer']) if 'buffer' in region else 'no buffer'}"
            f", shape={Path(region['shape']['shape']).name if isinstance(region['shape']['shape'], (str, Path)) else 'gpd.GeoDataFrame'}"
            f" - clisops v{clisops.__version__}"
        )

    else:
        raise ValueError("Subsetting type not recognized")

    history = (
        new_history + " \n " + ds_subset.attrs["history"]
        if "history" in ds_subset.attrs
        else new_history
    )
    ds_subset.attrs["history"] = history

    return ds_subset
