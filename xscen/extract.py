# noqa: D100
import datetime
import logging
import os
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import xclim as xc
from intake_esm.derived import DerivedVariableRegistry

from .catalog import DataCatalog  # noqa
from .catalog import ID_COLUMNS, concat_data_catalogs, generate_id, subset_file_coverage
from .catutils import parse_from_ds
from .config import parse_config
from .indicators import load_xclim_module, registry_from_module
from .spatial import subset
from .utils import CV
from .utils import ensure_correct_time as _ensure_correct_time
from .utils import get_cat_attrs, natural_sort, standardize_periods

logger = logging.getLogger(__name__)


__all__ = [
    "extract_dataset",
    "resample",
    "search_data_catalogs",
    "get_warming_level",
    "subset_warming_level",
]


def clisops_subset(ds: xr.Dataset, region: dict) -> xr.Dataset:
    """Customize a call to clisops.subset() that allows for an automatic buffer around the region.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to be subsetted
    region : dict
        Description of the region and the subsetting method (required fields listed in the Notes)

    Notes
    -----
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
        Subsetted Dataset.

    See Also
    --------
    clisops.core.subset.subset_gridpoint, clisops.core.subset.subset_bbox, clisops.core.subset.subset_shape
    """
    warnings.warn(
        "clisops_subset is deprecated and will not be available in future versions. "
        "Use xscen.spatial.subset instead.",
        category=FutureWarning,
    )

    ds_subset = subset(ds, region=region)

    return ds_subset


@parse_config
def extract_dataset(
    catalog: DataCatalog,
    *,
    variables_and_freqs: dict = None,
    periods: list = None,
    region: Optional[dict] = None,
    to_level: str = "extracted",
    ensure_correct_time: bool = True,
    xr_open_kwargs: dict = None,
    xr_combine_kwargs: dict = None,
    preprocess: Callable = None,
    resample_methods: Optional[dict] = None,
    mask: Union[bool, xr.Dataset, xr.DataArray] = False,
) -> Union[dict, xr.Dataset]:
    """Take one element of the output of `search_data_catalogs` and returns a dataset, performing conversions and resampling as needed.

    Nothing is written to disk within this function.

    Parameters
    ----------
    catalog : DataCatalog
        Sub-catalog for a single dataset, one value of the output of `search_data_catalogs`.
    variables_and_freqs : dict
        Variables and freqs, following a 'variable: xrfreq-compatible str' format. A list of strings can also be provided.
        If None, it will be read from catalog._requested_variables and catalog._requested_variable_freqs
        (set by `variables_and_freqs` in `search_data_catalogs`)
    periods : list
        Either [start, end] or list of [start, end] for the periods to be evaluated.
        Will be read from catalog._requested_periods if None. Leave both None to extract everything.
    region : dict, optional
        Description of the region and the subsetting method (required fields listed in the Notes) used in `xscen.spatial.subset`.
    to_level : str
        The processing level to assign to the output.
        Defaults to 'extracted'
    ensure_correct_time : bool
        When True (default), even if the data has the correct frequency, its time coordinate is
        checked so that it exactly matches the frequency code (xrfreq). For example, daily data given at
        noon would be transformed to be given at midnight. If the time coordinate is invalid,
        it raises an error.
    xr_open_kwargs : dict, optional
        A dictionary of keyword arguments to pass to `DataCatalogs.to_dataset_dict`, which
        will be passed to `xr.open_dataset`.
    xr_combine_kwargs : dict, optional
        A dictionary of keyword arguments to pass to `DataCatalogs.to_dataset_dict`, which
        will be passed to `xr.combine_by_coords`.
    preprocess : callable, optional
        If provided, call this function on each dataset prior to aggregation.
    resample_methods : dict, optional
        Dictionary where the keys are the variables and the values are the resampling method.
        Options for the resampling method are {'mean', 'min', 'max', 'sum', 'wind_direction'}.
        If the method is not given for a variable, it is guessed from the variable name and frequency,
        using the mapping in CVs/resampling_methods.json. If the variable is not found there,
        "mean" is used by default.
    mask: xr.Dataset, bool
        A mask that is applied to all variables and only keeps data where it is True.
        Where the mask is False, variable values are replaced by NaNs.
        The mask should have the same dimensions as the variables extracted.
        If `mask` is a dataset, the dataset should have a variable named 'mask'.
        If `mask` is True, it will expect a `mask` variable at xrfreq `fx` to have been extracted.

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
            ['gridpoint', 'bbox', shape', 'sel']
        tile_buffer: float, optional
            Multiplier to apply to the model resolution.
        kwargs
            Arguments specific to the method used.

    See Also
    --------
    intake_esm.core.esm_datastore.to_dataset_dict, xarray.open_dataset, xarray.combine_by_coords
    """
    resample_methods = resample_methods or {}

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
            variables_and_freqs = defaultdict(list)
            for a, b in zip(
                catalog._requested_variables_true, catalog._requested_variable_freqs
            ):
                variables_and_freqs[a].extend([b])
        except ValueError:
            raise ValueError("Failed to determine the requested variables and freqs.")
    else:
        # Make everything a list
        variables_and_freqs = {
            k: [v] if not isinstance(v, list) else v
            for k, v in variables_and_freqs.items()
        }

    # Default arguments to send xarray
    xr_open_kwargs = xr_open_kwargs or {}
    xr_combine_kwargs = xr_combine_kwargs or {}
    xr_combine_kwargs.setdefault("data_vars", "minimal")

    # Open the catalog
    ds_dict = catalog.to_dataset_dict(
        xarray_open_kwargs=xr_open_kwargs,
        xarray_combine_by_coords_kwargs=xr_combine_kwargs,
        preprocess=preprocess,
        # Only print a progress bar when it is minimally useful
        progressbar=(len(catalog.keys()) > 1),
    )

    out_dict = {}
    for xrfreq in pd.unique([x for y in variables_and_freqs.values() for x in y]):
        ds = xr.Dataset()
        attrs = {}
        # iterate on the datasets, in reverse timedelta order
        for key, ds_ts in sorted(
            ds_dict.items(),
            key=lambda kv: pd.Timedelta(
                CV.xrfreq_to_timedelta(catalog[kv[0]].df.xrfreq.iloc[0], default="NAN")
            ),
            reverse=True,
        ):
            if "time" in ds_ts:
                if pd.Timedelta(
                    CV.xrfreq_to_timedelta(catalog[key].df.xrfreq.iloc[0])
                ) > pd.Timedelta(CV.xrfreq_to_timedelta(xrfreq)):
                    continue
                if ensure_correct_time:
                    # Expected freq (xrfreq is the wanted freq)
                    expfreq = catalog[key].df.xrfreq.iloc[0]
                    ds_ts = _ensure_correct_time(ds_ts, expfreq)

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
                    or xrfreq not in variables_and_freqs.get(var_name, [])
                    or var_name not in catalog._requested_variables_true
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

                if "time" not in da.dims or (
                    catalog[key].df["xrfreq"].iloc[0] == xrfreq
                ):
                    ds = ds.assign({var_name: da})
                else:  # check if it needs resampling
                    if pd.to_timedelta(
                        CV.xrfreq_to_timedelta(catalog[key].df["xrfreq"].iloc[0])
                    ) < pd.to_timedelta(CV.xrfreq_to_timedelta(xrfreq)):
                        logger.info(
                            f"Resampling {var_name} from [{catalog[key].df['xrfreq'].iloc[0]}]"
                            f" to [{xrfreq}]."
                        )
                        ds = ds.assign(
                            {
                                var_name: resample(
                                    da,
                                    xrfreq,
                                    ds=ds_ts,
                                    method=resample_methods.get(var_name, None),
                                )
                            }
                        )
                    else:
                        raise ValueError(
                            "Variable is at a coarser frequency than requested."
                        )

                attrs = ds_ts.attrs

        ds.attrs = attrs
        if "time" not in ds.dims:
            ds.attrs["cat:frequency"] = "fx"
            ds.attrs["cat:xrfreq"] = "fx"
        else:
            ds.attrs["cat:xrfreq"] = xrfreq
            ds.attrs["cat:frequency"] = CV.xrfreq_to_frequency(xrfreq)

        # Subset time on the periods
        periods_extract = deepcopy(periods)
        if periods_extract is None and hasattr(catalog, "_requested_periods"):
            periods_extract = catalog._requested_periods
        if periods_extract is not None and "time" in ds:
            periods_extract = standardize_periods(periods_extract)
            slices = []
            for period in periods_extract:
                slices.extend([ds.sel({"time": slice(period[0], period[1])})])
            ds = xr.concat(slices, dim="time", **xr_combine_kwargs)

        # subset to the region
        if region is not None:
            if (region["method"] in region) and (
                isinstance(region[region["method"]], dict)
            ):
                warnings.warn(
                    "You seem to be using a deprecated version of region. Please use the new formatting.",
                    category=FutureWarning,
                )
                region = deepcopy(region)
                if "buffer" in region:
                    region["tile_buffer"] = region.pop("buffer")
                _kwargs = region.pop(region["method"])
                region.update(_kwargs)

            ds = subset(ds, **region)

        # add relevant attrs
        ds.attrs["cat:processing_level"] = to_level
        if "cat:variable" not in ds.attrs:
            ds.attrs["cat:variable"] = parse_from_ds(ds, ["variable"])["variable"]

        out_dict[xrfreq] = ds

    if mask:
        if isinstance(mask, xr.Dataset):
            ds_mask = mask["mask"]
        elif isinstance(mask, xr.DataArray):
            ds_mask = mask
        elif (
            "fx" in out_dict and "mask" in out_dict["fx"]
        ):  # get mask that was extracted above
            ds_mask = out_dict["fx"]["mask"].copy()
        else:
            raise ValueError(
                "No mask found. Either pass a xr.Dataset/xr.DataArray to the `mask` argument or pass a `dc` that includes a dataset with a variable named `mask`."
            )

        # iter over all xrfreq to apply the mask
        for xrfreq, ds in out_dict.items():
            out_dict[xrfreq] = ds.where(ds_mask)
            if xrfreq == "fx":  # put back the mask
                out_dict[xrfreq]["mask"] = ds_mask

    return out_dict


def resample(
    da: xr.DataArray,
    target_frequency: str,
    *,
    ds: Optional[xr.Dataset] = None,
    method: Optional[str] = None,
) -> xr.DataArray:
    """Aggregate variable to the target frequency.

    Parameters
    ----------
    da  : xr.DataArray
        DataArray of the variable to resample, must have a "time" dimension and be of a
        finer temporal resolution than "target_timestep".
    target_frequency : str
        The target frequency/freq str, must be one of the frequency supported by pandas.
    ds : xr.Dataset, optional
        The "wind_direction" resampling method needs extra variables, which can be given here.
    method : {'mean', 'min', 'max', 'sum', 'wind_direction'}, optional
        The resampling method. If None (default), it is guessed from the variable name and frequency,
        using the mapping in CVs/resampling_methods.json. If the variable is not found there,
        "mean" is used by default.

    Returns
    -------
    xr.DataArray
        Resampled variable

    """
    var_name = da.name

    initial_frequency = xr.infer_freq(da.time.dt.round("T")) or "undetected"
    initial_frequency_td = pd.Timedelta(
        CV.xrfreq_to_timedelta(xr.infer_freq(da.time.dt.round("T")), None)
    )
    if initial_frequency_td == pd.Timedelta("1D"):
        logger.warning(
            "You appear to be resampling daily data using extract_dataset. "
            "It is advised to use compute_indicators instead, as it is far more robust."
        )
    elif initial_frequency_td > pd.Timedelta("1D"):
        logger.warning(
            "You appear to be resampling data that is coarser than daily. "
            "Be aware that this is not currently explicitely supported by xscen and might result in erroneous manipulations."
        )

    if method is None:
        if (
            target_frequency in CV.resampling_methods.dict
            and var_name in CV.resampling_methods.dict[target_frequency]
        ):
            method = CV.resampling_methods(target_frequency)[var_name]
            logger.info(
                f"Resampling method for {var_name}: '{method}', based on variable name and frequency."
            )

        elif var_name in CV.resampling_methods.dict["any"]:
            method = CV.resampling_methods("any")[var_name]
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


@parse_config
def search_data_catalogs(
    data_catalogs: Union[
        Union[str, os.PathLike], list[Union[str, os.PathLike]], DataCatalog
    ],
    variables_and_freqs: dict,
    *,
    other_search_criteria: Optional[dict] = None,
    exclusions: dict = None,
    match_hist_and_fut: bool = False,
    periods: list = None,
    coverage_kwargs: dict = None,
    id_columns: Optional[list[str]] = None,
    allow_resampling: bool = False,
    allow_conversion: bool = False,
    conversion_yaml: str = None,
    restrict_resolution: str = None,
    restrict_members: dict = None,
    restrict_warming_level: Union[dict, bool] = None,
) -> dict:
    """Search through DataCatalogs.

    Parameters
    ----------
    data_catalogs : Union[Union[str, os.PathLike], List[Union[str, os.PathLike]], DataCatalog]
        DataCatalog (or multiple, in a list) or paths to JSON/CSV data catalogs. They must use the same columns and aggregation options.
    variables_and_freqs : dict
        Variables and freqs to search for, following a 'variable: xr-freq-compatible-str' format. A list of strings can also be provided.
    other_search_criteria : dict, optional
        Other criteria to search for in the catalogs' columns, following a 'column_name: list(subset)' format.
    exclusions : dict, optional
        Same as other_search_criteria, but for eliminating results.
    match_hist_and_fut: bool, optional
        If True, historical and future simulations will be combined into the same line, and search results lacking one of them will be rejected.
    periods : list
        Either [start, end] or list of [start, end] for the periods to be evaluated.
    coverage_kwargs : dict
        Arguments to pass to subset_file_coverage (only used when periods is not None).
    id_columns : list, optional
        List of columns used to create a id column. If None is given, the original
        "id" is left.
    allow_resampling : bool
         If True (default), variables with a higher time resolution than requested are considered.
    allow_conversion : bool
        If True (default) and if the requested variable cannot be found, intermediate variables are
        searched given that there exists a converting function in the "derived variable registry".
    conversion_yaml : str
        Path to a YAML file that defines the possible conversions (used alongside 'allow_conversion'=True).
        This file should follow the xclim conventions for building a virtual module.
        If None, the "derived variable registry" will be defined by the file in "xscen/xclim_modules/conversions.yml"
    restrict_resolution : str
        Used to restrict the results to the finest/coarsest resolution available for a given simulation.
        ['finest', 'coarsest'].
    restrict_members : dict
        Used to restrict the results to a given number of members for a given simulation.
        Currently only supports {"ordered": int} format.
    restrict_warming_level : bool, dict
        Used to restrict the results only to datasets that exist in the csv used to compute warming levels in `subset_warming_level`.
        If True, this will only keep the datasets that have a mip_era, source, experiment
        and member combination that exist in the csv. This does not guarantees that a given warming level will be reached, only that the datasets have corresponding columns in the csv.
        More option can be added by passing a dictionary instead of a boolean.
        If {'ignore_member':True}, it will disregard the member when trying to match the dataset to a column.
        If {tas_csv: Path_to_csv}, it will use an alternative csv instead of the default one provided by xscen.

    Notes
    -----
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

    See Also
    --------
    intake_esm.core.esm_datastore.search
    """
    cat_kwargs = {}
    if allow_conversion:
        if conversion_yaml is None:
            conversion_yaml = Path(__file__).parent / "xclim_modules" / "conversions"
        cat_kwargs = {
            "registry": registry_from_module(load_xclim_module(conversion_yaml))
        }

    # Cast paths to single item list
    if isinstance(data_catalogs, (str, Path)):
        data_catalogs = [data_catalogs]

    # Prepare a unique catalog to search from, with the DerivedCat added if required
    if isinstance(data_catalogs, DataCatalog):
        catalog = DataCatalog(
            {"esmcat": data_catalogs.esmcat.dict(), "df": data_catalogs.df},
            **cat_kwargs,
        )
        data_catalogs = [catalog]  # simply for a meaningful logging line
    elif isinstance(data_catalogs, list) and all(
        isinstance(dc, DataCatalog) for dc in data_catalogs
    ):
        catalog = DataCatalog(
            {
                "esmcat": data_catalogs[0].esmcat.dict(),
                "df": pd.concat([dc.df for dc in data_catalogs], ignore_index=True),
            },
            **cat_kwargs,
        )
    elif isinstance(data_catalogs, list) and all(
        isinstance(dc, str) for dc in data_catalogs
    ):
        data_catalogs = [
            DataCatalog(path) if path.endswith(".json") else DataCatalog.from_df(path)
            for path in data_catalogs
        ]
        catalog = DataCatalog(
            {
                "esmcat": data_catalogs[0].esmcat.dict(),
                "df": pd.concat([dc.df for dc in data_catalogs], ignore_index=True),
            },
            **cat_kwargs,
        )
    else:
        raise ValueError("Catalogs type not recognized.")
    logger.info(f"Catalog opened: {catalog} from {len(data_catalogs)} files.")

    if match_hist_and_fut:
        logger.info("Dispatching historical dataset to future experiments.")
        catalog = _dispatch_historical_to_future(catalog, id_columns)

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
    if restrict_warming_level:
        if isinstance(restrict_warming_level, bool):
            restrict_warming_level = {}
        restrict_warming_level.setdefault("ignore_member", False)
        restrict_warming_level.setdefault("tas_csv", None)
        catalog.esmcat._df = _restrict_wl(catalog.df, restrict_warming_level)

    if id_columns is not None or catalog.df["id"].isnull().any():
        ids = generate_id(catalog.df, id_columns)
        if id_columns is not None:
            # Recreate id from user specifications
            catalog.df["id"] = ids
        else:
            # Only fill in the missing IDs
            catalog.df["id"] = catalog.df["id"].fillna(ids)

    if catalog.df.empty:
        logger.warning("Found no match corresponding to the 'other' search criteria.")
        return {}

    coverage_kwargs = coverage_kwargs or {}
    periods = standardize_periods(periods)

    logger.info(f"Iterating over {len(catalog.unique('id'))} potential datasets.")
    # Loop on each dataset to assess whether they have all required variables
    # And select best freq/timedelta for each
    catalogs = {}
    if len(catalog) > 0:
        for (sim_id,), scat in catalog.iter_unique("id"):
            # Find all the entries that match search parameters
            varcats = []
            for var_id, xrfreqs in variables_and_freqs.items():
                if isinstance(xrfreqs, str):
                    xrfreqs = [xrfreqs]
                for xrfreq in xrfreqs:
                    if xrfreq == "fx":
                        varcat = scat.search(
                            xrfreq=xrfreq,
                            variable=var_id,
                            require_all_on=["id", "xrfreq"],
                        )
                        if len(varcat) == 0:
                            # Try searching in other experiments or members
                            scat_id = {
                                i: scat.df[i].iloc[0]
                                for i in id_columns or ID_COLUMNS
                                if i in scat.df.columns
                            }
                            scat_id.pop("experiment", None)
                            scat_id.pop("member", None)
                            varcat = catalog.search(
                                **scat_id,
                                xrfreq=xrfreq,
                                variable=var_id,
                                require_all_on=["id", "xrfreq"],
                            )
                            if len(varcat) > 1:
                                varcat.esmcat._df = varcat.df.iloc[[0]]
                            if len(varcat) == 1:
                                logger.warning(
                                    f"Dataset {sim_id} doesn't have the fixed field {var_id}, but it can be acquired from {varcat.df['id'].iloc[0]}."
                                )
                                for i in {"member", "experiment", "id"}.intersection(
                                    varcat.df.columns
                                ):
                                    varcat.df.loc[:, i] = scat.df[i].iloc[0]

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
                        varcat = scat.search(
                            variable=var_id, require_all_on=["id", "xrfreq"]
                        )
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
                            varcat.df.xrfreq.apply(
                                CV.xrfreq_to_timedelta, default="NAN"
                            )
                        )
                        # else is joker (any timedelta)
                        lower_frq = (
                            np.less(varcat.df.timedelta, td)
                            if pd.notnull(td)
                            else False
                        )
                        varcat.esmcat._df = varcat.df[
                            same_frq | (lower_frq & allow_resampling)
                        ]

                        # For each dataset (id - xrfreq - processing_level - domain - variable), make sure that file availability covers the requested time periods
                        if periods is not None and len(varcat) > 0:
                            valid_tp = []
                            for var, group in varcat.df.groupby(
                                varcat.esmcat.aggregation_control.groupby_attrs
                                + ["variable"]
                            ):
                                valid_tp.append(
                                    subset_file_coverage(
                                        group, periods, **coverage_kwargs
                                    )
                                )  # If valid, this returns the subset of files that cover the time period
                            varcat.esmcat._df = pd.concat(valid_tp)

                        # We now select the coarsest timedelta for each variable
                        # we need to re-iterate over variables in case we used the registry (and thus there are multiple variables in varcat)
                        rows = []
                        for var, group in varcat.df.groupby("variable"):
                            rows.append(group[group.timedelta == group.timedelta.max()])
                        if rows:
                            # check if the requested variable exists and if so, remove DeriveVariable references
                            v_list = [
                                rows[i]["variable"].iloc[0] for i in range(len(rows))
                            ]
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
                        logger.debug(
                            f"Dataset {sim_id} doesn't have all needed variables (missing at least {var_id})."
                        )
                        break
                    if "timedelta" in varcat.df.columns:
                        varcat.df.drop(columns=["timedelta"], inplace=True)
                    varcat._requested_variable_freqs = [xrfreq]
                    varcats.append(varcat)

                else:
                    continue
                break
            else:
                catalogs[sim_id] = concat_data_catalogs(*varcats)
                if periods is not None:
                    catalogs[sim_id]._requested_periods = periods

    if len(catalogs) > 0:
        logger.info(
            f"Found {len(catalogs)} with all variables requested and corresponding to the criteria."
        )
    else:
        logger.warning("Found no match corresponding to the search criteria.")

    if restrict_resolution is not None and len(catalogs) > 0:
        catalogs = _restrict_by_resolution(catalogs, id_columns, restrict_resolution)

    if restrict_members is not None and len(catalogs) > 0:
        catalogs = _restrict_multimembers(catalogs, id_columns, restrict_members)

    return catalogs


@parse_config
def get_warming_level(
    realization: Union[xr.Dataset, dict, str, list],
    wl: float,
    *,
    window: int = 20,
    tas_baseline_period: list = None,
    ignore_member: bool = False,
    tas_csv: str = None,
    return_horizon: bool = True,
):
    """Use the IPCC Atlas method to return the window of time over which the requested level of global warming is first reached.

    Parameters
    ----------
    realization : xr.Dataset, dict, str or list of those
       Model to be evaluated. Needs the four fields mip_era, source, experiment and member,
       as a dict or in a Dataset's attributes. Strings should follow this formatting: {mip_era}_{source}_{experiment}_{member}.
       Lists of dicts, strings or Datasets are also accepted, in which case the output will be a dict.
       Regex wildcards (.*) are accepted, but may lead to unexpected results.
       Datasets should include the catalogue attributes (starting by "cat:") required to create such a string: 'cat:mip_era', 'cat:experiment',
       'cat:member', and either 'cat:source' for global models or 'cat:driving_model' for regional models.
       e.g. 'CMIP5_CanESM2_rcp85_r1i1p1'
    wl : float
       Warming level.
       e.g. 2 for a global warming level of +2 degree Celsius above the mean temperature of the `tas_baseline_period`.
    window : int
       Size of the rolling window in years over which to compute the warming level.
    tas_baseline_period : list
       [start, end] of the base period. The warming is calculated with respect to it. The default is ["1850", "1900"].
    ignore_member : bool
       Only used for Datasets. Decides whether to ignore the member when searching for the model run in tas_csv.
    tas_csv : str
       Path to a csv of annual global mean temperature with a row for each year and a column for each dataset.
       If None, it will default to data/IPCC_annual_global_tas.csv which was built from
       the IPCC atlas data from  Iturbide et al., 2020 (https://doi.org/10.5194/essd-12-2959-2020)
       and extra data from pilot models of MRCC5 and ClimEx.
    return_horizon: bool
        If True, the output will be a list following the format ['start_yr', 'end_yr']
        If False, the output will be a string representing the middle of the period.

    Returns
    -------
    dict, list or str
        If `realization` is a Dataset, a dict or a string, the output will follow the format indicated by `return_period`.
        If `realization` is a list, the output will be a dictionary where the keys are the selected columns from the csv and the values follow the format indicated by `return_period`.
    """
    tas_baseline_period = standardize_periods(
        tas_baseline_period or ["1850", "1900"], multiple=False
    )

    if (window % 2) not in {0, 1}:
        raise ValueError(f"window should be an integer, received {type(window)}")

    FIELDS = ["mip_era", "source", "experiment", "member"]

    if tas_csv is None:
        tas_csv = Path(__file__).parent / "data" / "IPCC_annual_global_tas.csv"

    if isinstance(realization, (xr.Dataset, str, dict)):
        realization = [realization]
    info_models = []
    for real in realization:
        info = {}
        if isinstance(real, xr.Dataset):
            attrs = get_cat_attrs(real)
            # get info on ds
            if attrs.get("driving_model") is None:
                info["source"] = attrs["source"]
            else:
                info["source"] = attrs["driving_model"]
            info["experiment"] = attrs["experiment"]
            info["member"] = ".*" if ignore_member else attrs["member"]
            info["mip_era"] = attrs["mip_era"]
        elif isinstance(real, str):
            (
                info["mip_era"],
                info["source"],
                info["experiment"],
                info["member"],
            ) = real.split("_")
        elif isinstance(real, dict) and set(real.keys()).issuperset(
            (set(FIELDS) - {"member"}) if ignore_member else FIELDS
        ):
            info = real
            if ignore_member:
                info["member"] = ".*"
        else:
            raise ValueError(
                f"'realization' must be a Dataset, dict, string or list. Received {type(real)}."
            )
        info_models.append(info)

    # open csv, split column names for easier usage
    annual_tas = pd.read_csv(tas_csv, index_col="year")
    models = pd.DataFrame.from_records(
        [c.split("_") for c in annual_tas.columns],
        index=annual_tas.columns,
        columns=FIELDS,
    )

    out = {}
    for model in info_models:
        # choose colum based in ds cat attrs
        mip = models.mip_era.str.match(model["mip_era"])
        src = models.source.str.match(model["source"])
        if not src.any():
            # Maybe it's an RCM, then source may contain the institute
            src = models.source.apply(lambda s: model["source"].endswith(s))
        exp = models.experiment.str.match(model["experiment"])
        mem = models.member.str.match(model["member"])

        candidates = models[mip & src & exp & mem]
        if candidates.empty:
            warnings.warn(
                f"No columns fit the attributes of the input dataset ({model})."
            )
            selected = "_".join([model[c] for c in FIELDS])
            out[selected] = [None, None] if return_horizon else None
            continue
        if len(candidates) > 1:
            logger.info(
                "More than one column of the csv fits the dataset metadata. Choosing the first one."
            )
        selected = candidates.index[0]
        right_column = annual_tas.loc[:, selected]

        logger.debug(
            f"Computing warming level +{wl}°C for {model} from column: {selected}."
        )

        # compute reference temperature for the warming
        mean_base = right_column.loc[
            tas_baseline_period[0] : tas_baseline_period[1]
        ].mean()

        yearly_diff = right_column - mean_base  # difference from reference

        # get the start and end date of the window when the warming level is first reached
        # shift(-1) is needed to reproduce IPCC results.
        # rolling defines the window as [n-10,n+9], but the the IPCC defines it as [n-9,n+10], where n is the center year.
        if window % 2 == 0:  # Even window
            rolling_diff = (
                yearly_diff.rolling(window=window, min_periods=window, center=True)
                .mean()
                .shift(-1)
            )
        else:  # window % 2 == 1:  # Odd windows do not require the shift
            rolling_diff = yearly_diff.rolling(
                window=window, min_periods=window, center=True
            ).mean()

        yr = rolling_diff.where(rolling_diff >= wl).first_valid_index()
        if yr is None:
            logger.info(
                f"Global warming level of +{wl}C is not reached by the last year of the provided 'tas_csv' file for {selected}."
            )
            out[selected] = [None, None] if return_horizon else None
        else:
            start_yr = int(yr - window / 2 + 1)
            end_yr = int(yr + window / 2)
            out[selected] = (
                standardize_periods([start_yr, end_yr], multiple=False)
                if return_horizon
                else str(yr)
            )

    if len(out) != len(realization):
        warnings.warn(
            "Two or more input model specifications pointed towards the same column in the CSV, "
            "the length of the output is different from the input."
        )
    if len(realization) == 1:
        out = out.popitem()[1]
    return out


@parse_config
def subset_warming_level(
    ds: xr.Dataset,
    wl: float,
    to_level: str = "warminglevel-{wl}vs{period0}-{period1}",
    wl_dim: str = "+{wl}Cvs{period0}-{period1}",
    **kwargs,
):
    """Subsets the input dataset with only the window of time over which the requested level of global warming is first reached, using the IPCC Atlas method.

    Parameters
    ----------
    ds : xr.Dataset
       Input dataset.
       The dataset should include attributes to help recognize it and find its
       warming levels - 'cat:mip_era', 'cat:experiment', 'cat:member', and either
       'cat:source' for global models or 'cat:driving_institution' (optional) + 'cat:driving_model' for regional models.
    wl : float
       Warming level.
       eg. 2 for a global warming level of +2 degree Celsius above the mean temperature of the `tas_baseline_period`.
    to_level :
       The processing level to assign to the output.
       Use "{wl}", "{period0}" and "{period1}" in the string to dynamically include
       `wl`, 'tas_baseline_period[0]' and 'tas_baseline_period[1]'.
    wl_dim : str
       The value to use to fill the new `warminglevel` dimension.
       Use "{wl}", "{period0}" and "{period1}" in the string to dynamically include
       `wl`, 'tas_baseline_period[0]' and 'tas_baseline_period[1]'.
       If None, no new dimensions will be added.

    kwargs
        Instructions on how to search for warming levels.
        The keyword arguments are passed to `get_warming_level()`

        Valid keyword aguments are:
            window : int
            tas_baseline_period : list
            ignore_member : bool
            tas_csv : str
            return_horizon: bool

    Returns
    -------
    xr.Dataset
        Warming level dataset.
    """
    start_yr, end_yr = get_warming_level(ds, wl=wl, return_horizon=True, **kwargs)

    if start_yr is None:
        return None
    elif any(yr not in ds.time.dt.year for yr in range(int(start_yr), int(end_yr) + 1)):
        logger.info(
            f"{ds.attrs.get('cat:id', 'The provided dataset')} does not sufficiently cover the time interval for +{wl}°C ({start_yr}, {end_yr})."
        )
        return None

    # cut the window selected above
    ds_wl = ds.sel(time=slice(start_yr, end_yr))

    tas_baseline_period = kwargs.get("tas_baseline_period", ["1850", "1900"])
    if wl_dim:
        ds_wl = ds_wl.expand_dims(
            dim={
                "warminglevel": [
                    wl_dim.format(
                        wl=wl,
                        period0=tas_baseline_period[0],
                        period1=tas_baseline_period[1],
                    )
                ]
            },
            axis=0,
        )
        ds_wl.warminglevel.attrs[
            "baseline"
        ] = f"{tas_baseline_period[0]}-{tas_baseline_period[1]}"

    if to_level is not None:
        ds_wl.attrs["cat:processing_level"] = to_level.format(
            wl=wl,
            period0=tas_baseline_period[0],
            period1=tas_baseline_period[1],
        )

    return ds_wl


def _dispatch_historical_to_future(catalog: DataCatalog, id_columns: list):
    """Update a DataCatalog by recopying each "historical" entry to its corresponding future experiments.

    For examples, if an historical entry has corresponding "ssp245" and "ssp585" entries,
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

                # Remove fixed fields that already exist in the future experiment
                dupes = pd.concat(
                    [
                        exp_hist.loc[exp_hist["frequency"] == "fx"],
                        sub_sdf.loc[
                            (sub_sdf["frequency"] == "fx")
                            & (sub_sdf["experiment"] == exp_id)
                        ],
                    ]
                ).duplicated(
                    subset=[
                        "same_hist_member",
                        "variable",
                        "frequency",
                        "experiment",
                        "activity",
                    ],
                    keep=False,
                )
                dupes = dupes[dupes]  # Only keep the duplicated rows
                exp_hist = exp_hist.loc[exp_hist.index.difference(dupes.index)]

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


def _restrict_by_resolution(catalogs: dict, id_columns: list, restrictions: str):
    """Update the results from search_data_catalogs by removing simulations with multiple resolutions available.

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
            if "MIP" in pd.unique(df_sim["activity"])[0]:
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

            # Note: For CORDEX, the order is dictated by both the grid label
            # and the resolution itself as well as the domain name
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
                    f"Dataset {i} seems to have multiple resolutions, "
                    "but its activity is not yet recognized or supported."
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


def _restrict_multimembers(catalogs: dict, id_columns: list, restrictions: dict):
    """Update the results from search_data_catalogs by removing simulations with multiple members available.

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


def _restrict_wl(df, restrictions: dict):
    """Update the results from search_data_catalogs by removing simulations that are not available in the warming level csv."""
    tas_csv = restrictions["tas_csv"]
    if tas_csv is None:
        tas_csv = Path(__file__).parent / "data/IPCC_annual_global_tas.csv"

    # open csv
    annual_tas = pd.read_csv(tas_csv, index_col="year")

    if restrictions["ignore_member"]:
        df["csv_name"] = df["mip_era"].str.cat(
            [df["source"], df["experiment"]], sep="_"
        )
        csv_source = ["_".join(x.split("_")[:-1]) for x in annual_tas.columns[1:]]
    else:
        df["csv_name"] = df["mip_era"].str.cat(
            [df["source"], df["experiment"], df["member"]], sep="_"
        )
        csv_source = list(annual_tas.columns[1:])

    to_keep = df["csv_name"].isin(csv_source)
    removed = pd.unique(df[~to_keep]["id"])

    df = df[to_keep]
    logger.info(
        f"Removing the following datasets because of the restriction for warming levels: {list(removed)}"
    )

    df = df.drop(columns=["csv_name"])

    return df
