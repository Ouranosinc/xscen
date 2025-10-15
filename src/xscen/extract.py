"""Functions to find and extract data from a catalog."""

import datetime
import logging
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xclim as xc
from intake_esm.derived import DerivedVariableRegistry
from scipy.interpolate import interp1d
from xclim.core.calendar import compare_offsets


try:
    import xclim.indicators.convert as convert
except ImportError:  # FIXME: Remove when we pin xclim >= 0.58
    import xclim.indicators.atmos as convert

from .catalog import (
    ID_COLUMNS,
    DataCatalog,
    concat_data_catalogs,
    generate_id,
    subset_file_coverage,
)
from .catutils import parse_from_ds
from .config import parse_config
from .indicators import load_xclim_module, registry_from_module
from .spatial import subset
from .utils import CV, _xarray_defaults, get_cat_attrs, natural_sort, standardize_periods, xrfreq_to_timedelta
from .utils import ensure_correct_time as _ensure_correct_time


logger = logging.getLogger(__name__)


__all__ = [
    "extract_dataset",
    "get_period_from_warming_level",
    "get_warming_level",
    "get_warming_level_from_period",
    "resample",
    "search_data_catalogs",
    "subset_warming_level",
]


@parse_config
def extract_dataset(  # noqa: C901
    catalog: DataCatalog,
    *,
    variables_and_freqs: dict | None = None,
    periods: list[str | int] | list[list[str | int]] | None = None,
    region: dict | None = None,
    to_level: str = "extracted",
    ensure_correct_time: bool = True,
    xr_open_kwargs: dict | None = None,
    xr_combine_kwargs: dict | None = None,
    preprocess: Callable | None = None,
    resample_methods: dict | None = None,
    mask: bool | xr.Dataset | xr.DataArray = False,
) -> dict:
    """
    Take one element of the output of `search_data_catalogs` and returns a dataset,
    performing conversions and resampling as needed.

    Nothing is written to disk within this function.

    Parameters
    ----------
    catalog : DataCatalog
        Sub-catalog for a single dataset, one value of the output of `search_data_catalogs`.
    variables_and_freqs : dict, optional
        Variables and frequencies, following a 'variable: xrfreq-compatible str' format. A list of strings can also be provided.
        If None, it will be read from `catalog._requested_variables` and `catalog._requested_variable_freqs`
        (set by `variables_and_freqs` in `search_data_catalogs`).
    periods : list of str or list of int or list of lists of str or list of lists of int, optional
        Either [start, end] or list of [start, end] for the periods to be evaluated.
        Will be read from catalog._requested_periods if None. Leave both None to extract everything.
    region : dict, optional
        Description of the region and the subsetting method (required fields listed in the Notes) used in `xscen.spatial.subset`.
    to_level : str
        The processing level to assign to the output. Defaults to 'extracted'.
    ensure_correct_time : bool
        When True (default), even if the data has the correct frequency, its time coordinate is
        checked so that it exactly matches the frequency code (xrfreq).
        For example, daily data given at noon would be transformed to be given at midnight.
        If the time coordinate is invalid, it raises an error.
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
        using the mapping in CVs/resampling_methods.json.
        If the variable is not found there, "mean" is used by default.
    mask : xr.Dataset or xr.DataArray or bool
        A mask that is applied to all variables and only keeps data where it is True.
        Where the mask is False, variable values are replaced by NaNs.
        The mask should have the same dimensions as the variables extracted.
        If `mask` is a dataset, the dataset should have a variable named 'mask'.
        If `mask` is True, it will expect a `mask` variable at xrfreq `fx` to have been extracted.

    Returns
    -------
    dict
        Dictionary (keys = xrfreq) with datasets containing all available and computed variables,
        subsetted to the region, everything resampled to the requested frequency.

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
        raise ValueError("The extraction catalog must have a unique id and unique processing_level.")
    if region is None and len(unique["domain"]) > 1:
        raise ValueError("If a subset region is not given, the extraction catalog must have a unique domain.")

    if variables_and_freqs is None:
        try:
            variables_and_freqs = defaultdict(list)
            for a, b in zip(catalog._requested_variables_true, catalog._requested_variable_freqs, strict=False):
                variables_and_freqs[a].extend([b])
        except ValueError as err:
            raise ValueError("Failed to determine the requested variables and freqs.") from err
    else:
        # Make everything a list
        variables_and_freqs = {k: [v] if not isinstance(v, list) else v for k, v in variables_and_freqs.items()}

    # Default arguments to send xarray
    xr_kwargs = _xarray_defaults(xr_open_kwargs=xr_open_kwargs or {}, xr_combine_kwargs=xr_combine_kwargs or {})

    # Open the catalog
    ds_dict = catalog.to_dataset_dict(
        preprocess=preprocess,
        # Only print a progress bar when it is minimally useful
        progressbar=(len(catalog.keys()) > 1),
        **xr_kwargs,
    )

    out_dict = {}
    for xrfreq in np.unique([x for y in variables_and_freqs.values() for x in y]):
        ds = xr.Dataset()
        attrs = {}
        # iterate on the datasets, in reverse timedelta order
        for key, ds_ts in sorted(
            ds_dict.items(),
            key=lambda kv: pd.Timedelta(CV.xrfreq_to_timedelta(catalog[kv[0]].df.xrfreq.iloc[0], default="NAN")),
            reverse=True,
        ):
            if "time" in ds_ts:
                if pd.Timedelta(CV.xrfreq_to_timedelta(catalog[key].df.xrfreq.iloc[0])) > pd.Timedelta(CV.xrfreq_to_timedelta(xrfreq)):
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
                if var_name in ds or xrfreq not in variables_and_freqs.get(var_name, []) or var_name not in catalog._requested_variables_true:
                    continue

                # TODO: This is a temporary fix for an xclim bug where the grid_mapping attribute is not transferred upon calculation
                if var_name in catalog.derivedcat and len(set(ds_ts.data_vars).intersection(catalog.derivedcat[var_name].query["variable"])) >= 1:
                    grid_mapping = np.unique(
                        [
                            ds_ts[v].attrs["grid_mapping"]
                            for v in set(ds_ts.data_vars).intersection(catalog.derivedcat[var_name].query["variable"])
                            if "grid_mapping" in ds_ts[v].attrs
                        ]
                    )
                    if len(grid_mapping) == 1:
                        da.attrs["grid_mapping"] = grid_mapping[0]
                    elif len(grid_mapping) > 1:
                        raise ValueError("Multiple grid_mapping detected.")

                if "time" not in da.dims or (catalog[key].df["xrfreq"].iloc[0] == xrfreq):
                    ds = ds.assign({var_name: da})
                else:  # check if it needs resampling
                    if pd.to_timedelta(CV.xrfreq_to_timedelta(catalog[key].df["xrfreq"].iloc[0])) < pd.to_timedelta(CV.xrfreq_to_timedelta(xrfreq)):
                        msg = f"Resampling {var_name} from [{catalog[key].df['xrfreq'].iloc[0]}] to [{xrfreq}]."
                        logger.info(msg)
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
                        raise ValueError("Variable is at a coarser frequency than requested.")

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
            ds = xr.concat(slices, dim="time", **xr_kwargs["xarray_combine_by_coords_kwargs"])

        # subset to the region
        if region is not None:
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
        elif "fx" in out_dict and "mask" in out_dict["fx"]:  # get mask that was extracted above
            ds_mask = out_dict["fx"]["mask"].copy()
        else:
            raise ValueError(
                "No mask found. Either pass a xr.Dataset/xr.DataArray to the `mask` argument "
                "or pass a `dc` that includes a dataset with a variable named `mask`."
            )

        # iter over all xrfreq to apply the mask
        for xrfreq, ds in out_dict.items():
            out_dict[xrfreq] = ds.where(ds_mask)
            if xrfreq == "fx":  # put back the mask
                out_dict[xrfreq]["mask"] = ds_mask

    return out_dict


def resample(  # noqa: C901
    da: xr.DataArray,
    target_frequency: str,
    *,
    ds: xr.Dataset | None = None,
    initial_frequency: str | None = None,
    method: str | None = None,
    missing: str | dict | None = None,
) -> xr.DataArray:
    """
    Aggregate variable to the target frequency.

    If the input frequency is greater than a week, the resampling operation is weighted by
    the number of days in each sampling period.

    Parameters
    ----------
    da : xr.DataArray
        DataArray of the variable to resample, must have a "time" dimension and be of a
        finer temporal resolution than "target_frequency".
    target_frequency : str
        The target frequency/freq str, must be one of the frequency supported by xarray.
    ds : xr.Dataset, optional
        The "wind_direction" resampling method needs extra variables, which can be given here.
    initial_frequency : str, optional
        The data frequency. It will be inferred, but can be given here instead if the data has missing time steps.
    method : {'mean', 'min', 'max', 'sum', 'wind_direction'}, optional
        The resampling method. If None (default), it is guessed from the variable name and frequency,
        using the mapping in CVs/resampling_methods.json. If the variable is not found there,
        "mean" is used by default.
    missing: {'mask', 'drop'} or dict, optional
        If 'mask' or 'drop', target periods that would have been computed from fewer timesteps than expected
        are masked or dropped, using a threshold of 5% of missing data.
        E.g. the first season of a `target_frequency` of "QS-DEC" will be masked or dropped if data starts in January.
        If a dict, points to a xclim check missing method which will mask periods according to the number of NaN values.
        The dict must contain a "method" field corresponding to the xclim method name and may contain
        any other args to pass. Options are documented in :py:mod:`xclim.core.missing`.

    Returns
    -------
    xr.DataArray
        Resampled variable
    """
    var_name = da.name

    if initial_frequency is None:
        initial_frequency = xr.infer_freq(da.time.dt.round("min")) or "undetected"
        if initial_frequency == "undetected":
            warnings.warn(
                "Could not infer the frequency of the dataset. "
                "Be aware that this might result in erroneous manipulations if the actual timestep is monthly or longer.",
                stacklevel=2,
            )
            if missing is not None:
                raise ValueError(
                    "Can't perform missing checks if the data's frequency is not inferable. "
                    "You can pass `initial_frequency` to avoid this, or disable missing checks with `missing=None`."
                )

    if method is None:
        if target_frequency in CV.resampling_methods.dict and var_name in CV.resampling_methods.dict[target_frequency]:
            method = CV.resampling_methods(target_frequency)[var_name]
            msg = f"Resampling method for {var_name}: '{method}', based on variable name and frequency."
            logger.info(msg)

        elif var_name in CV.resampling_methods.dict["any"]:
            method = CV.resampling_methods("any")[var_name]
            msg = f"Resampling method for {var_name}: '{method}', based on variable name."
            logger.info(msg)

        else:
            method = "mean"
            msg = f"Resampling method for {var_name} defaulted to: 'mean'."
            logger.info(msg)

    weights = None
    if (
        initial_frequency != "undetected"
        and compare_offsets(initial_frequency, ">", "W")
        and method in ["mean", "median", "std", "var", "wind_direction"]
    ):
        # More than a week -> non-uniform sampling length!
        t = xr.date_range(
            da.indexes["time"][0],
            periods=da.time.size + 1,
            freq=initial_frequency,
            calendar=da.time.dt.calendar,
        )
        # This is the number of days in each sampling period
        days_per_step = xr.DataArray(t, dims=("time",), coords={"time": t}).diff("time", label="lower").dt.days
        days_per_period = (
            days_per_step.resample(time=target_frequency)
            .sum()  # Total number of days per period
            .sel(time=days_per_step.time, method="ffill")  # Up-sample to initial freq
            .assign_coords(time=days_per_step.time)  # Not sure why we need this, but time coord is from the resample even after sel
        )
        weights = days_per_step / days_per_period

    # TODO : Support non-surface wind?
    if method == "wind_direction":
        ds[var_name] = da
        if ds is None or not any(
            [
                all(v in ds for v in ["uas", "vas"]),
                all(v in ds for v in ["sfcWind", "sfcWindfromdir"]),
            ]
        ):
            raise ValueError("Resampling method 'wind_direction' failed to find all required variables.")

        # The method requires uas, vas, and sfcWind. Acquire all of them.
        if all(v in ds for v in ["uas", "vas"]):
            uas, vas = ds.uas, ds.vas
        else:
            uas, vas = convert.wind_vector_from_speed(ds.sfcWind, ds.sfcWindfromdir)
        if "sfcWind" not in ds:
            ds["sfcWind"], _ = convert.wind_speed_from_vector(uas=ds["uas"], vas=ds["vas"])

        # Resample first to find the average wind speed and components
        if weights is not None:
            with xr.set_options(keep_attrs=True):
                ds = (ds * weights).resample(time=target_frequency).sum(dim="time")
        else:
            ds = ds.resample(time=target_frequency).mean(dim="time", keep_attrs=True)

        # Based on Vector Magnitude and Direction equations
        # For example: https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:vectors/x9e81a4f98389efdf:component-form/a/vector-magnitude-and-direction-review  # noqa: E501

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
            _, out = convert.wind_speed_from_vector(uas=uas, vas=vas)
        else:
            out = ds[var_name]

    elif weights is not None:
        if method == "mean":
            # Avoiding resample().map() is much more performant
            with xr.set_options(keep_attrs=True):
                out = (da * weights).resample(time=target_frequency).sum(dim="time").rename(da.name)
        else:
            kws = {"q": 0.5} if method == "median" else {}
            ds = xr.merge([da, weights.rename("weights")])
            out = ds.resample(time=target_frequency).map(
                lambda grp: getattr(
                    grp.drop_vars("weights").weighted(grp.weights),
                    method if method != "median" else "quantile",
                )(dim="time", **kws)
            )[da.name]
    else:
        out = getattr(da.resample(time=target_frequency), method)(dim="time", keep_attrs=True)

    missing_note = " "
    initial_td = xrfreq_to_timedelta(initial_frequency) if initial_frequency != "undetected" else None
    if missing in ["mask", "drop"] and not pd.isnull(initial_td):
        steps_per_period = xr.ones_like(da.time, dtype="int").resample(time=target_frequency).sum()
        t = xr.date_range(
            steps_per_period.indexes["time"][0],
            periods=steps_per_period.time.size + 1,
            freq=target_frequency,
        )

        expected = xr.DataArray(t, dims=("time",), coords={"time": t}).diff("time", label="lower") / initial_td
        complete = (steps_per_period / expected) > 0.95
        action = "masking" if missing == "mask" else "dropping"
        missing_note = f", {action} incomplete periods "
    elif isinstance(missing, dict):
        missmeth = missing.pop("method")
        miss = xc.core.missing.MISSING_METHODS[missmeth](**missing)
        complete = ~miss(da, target_frequency, initial_frequency)
        missing = "mask"
        missing_note = f", masking incomplete periods according to {miss} "
    if missing in {"mask", "drop"}:
        out = out.where(complete, drop=(missing == "drop"))

    new_history = (
        f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"{'weighted' if weights is not None else ''} {method} "
        f"resample from {initial_frequency} to {target_frequency}"
        f"{missing_note}- xarray v{xr.__version__}"
    )
    history = new_history + " \n " + out.attrs["history"] if "history" in out.attrs else new_history
    out.attrs["history"] = history

    return out


@parse_config
def search_data_catalogs(  # noqa: C901
    data_catalogs: (str | os.PathLike | DataCatalog | list[str | os.PathLike | DataCatalog]),
    variables_and_freqs: dict,
    *,
    other_search_criteria: dict | None = None,
    exclusions: dict | None = None,
    match_hist_and_fut: bool = False,
    periods: list[str] | list[list[str]] | None = None,
    coverage_kwargs: dict | None = None,
    id_columns: list[str] | None = None,
    allow_resampling: bool = False,
    allow_conversion: bool = False,
    conversion_yaml: str | None = None,
    restrict_resolution: str | None = None,
    restrict_members: dict | None = None,
    restrict_warming_level: dict | bool | None = None,
) -> dict:
    """
    Search through DataCatalogs.

    Parameters
    ----------
    data_catalogs : str, os.PathLike, DataCatalog, or a list of those
        DataCatalog (or multiple, in a list) or paths to JSON/CSV data catalogs.
        They must use the same columns and aggregation options.
    variables_and_freqs : dict
        Variables and freqs to search for, following a 'variable: xr-freq-compatible-str' format.
        A list of strings can also be provided.
    other_search_criteria : dict, optional
        Other criteria to search for in the catalogs' columns, following a 'column_name: list(subset)' format.
        You can also pass 'require_all_on: list(columns_name)' in order to only return results that correspond to
        all other criteria across the listed columns.
        More details available at https://intake-esm.readthedocs.io/en/stable/how-to/enforce-search-query-criteria-via-require-all-on.html .
    exclusions : dict, optional
        Same as other_search_criteria, but for eliminating results.
        Any result that matches any of the exclusions will be removed.
    match_hist_and_fut: bool
        If True, historical and future simulations will be combined into the same line,
        and search results lacking one of them will be rejected.
    periods : list of str or list of lists of str, optional
        Either [start, end] or list of [start, end] for the periods to be evaluated.
    coverage_kwargs : dict, optional
        Arguments to pass to subset_file_coverage (only used when periods is not None).
    id_columns : list, optional
        List of columns used to create a id column. If None is given, the original
        "id" is left.
    allow_resampling : bool
         If True (default), variables with a higher time resolution than requested are considered.
    allow_conversion : bool
        If True (default) and if the requested variable cannot be found, intermediate variables are
        searched given that there exists a converting function in the "derived variable registry".
    conversion_yaml : str, optional
        Path to a YAML file that defines the possible conversions (used alongside 'allow_conversion'=True).
        This file should follow the xclim conventions for building a virtual module.
        If None, the "derived variable registry" will be defined by the file in "xscen/xclim_modules/conversions.yml"
    restrict_resolution : str, optional
        Used to restrict the results to the finest/coarsest resolution available for a given simulation.
        ['finest', 'coarsest'].
    restrict_members : dict, optional
        Used to restrict the results to a given number of members for a given simulation.
        Currently only supports {"ordered": int} format.
    restrict_warming_level : bool or dict, optional
        Used to restrict the results only to datasets that exist in the csv used to compute warming levels in `subset_warming_level`.
        If True, this will only keep the datasets that have a mip_era, source, experiment and member combination that exist in the csv.
        This does not guarantee that a given warming level will be reached, only that the datasets have corresponding columns in the csv.
        More option can be added by passing a dictionary instead of a boolean.
        If {'ignore_member':True}, it will disregard the member when trying to match the dataset to a column.
        If {tas_src: Path_to_netcdf}, it will use an alternative netcdf instead of the default one provided by xscen.
        If 'wl' is a provided key, then `xs.get_period_from_warming_level` will be called
        and only datasets that reach the warming level will be kept.
        This can be combined with other arguments of the function, for example {'wl': 1.5, 'window': 30}.

    Notes
    -----
    - The "other_search_criteria" and "exclusions" arguments accept wildcard (*) and regular expressions.
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
    # Cast single items to a list
    if isinstance(data_catalogs, str | os.PathLike | DataCatalog):
        data_catalogs = [data_catalogs]

    # Open the catalogs given as paths
    data_catalogs = [
        (dc if not isinstance(dc, str | os.PathLike) else (DataCatalog(dc) if Path(dc).suffix == ".json" else DataCatalog.from_df(dc)))
        for dc in data_catalogs
    ]

    if not all(isinstance(dc, DataCatalog) for dc in data_catalogs):
        raise ValueError("Catalogs type not recognized.")

    cat_kwargs = {}
    if allow_conversion:
        if conversion_yaml is None:
            conversion_yaml = Path(__file__).parent / "xclim_modules" / "conversions"
        cat_kwargs = {"registry": registry_from_module(load_xclim_module(conversion_yaml))}

    # Prepare a unique catalog to search from, with the DerivedCat added if required
    catalog = DataCatalog(
        {
            "esmcat": data_catalogs[0].esmcat.model_dump(),
            "df": pd.concat([dc.df for dc in data_catalogs], ignore_index=True),
        },
        **cat_kwargs,
    )
    msg = f"Catalog opened: {catalog} from {len(data_catalogs)} files."
    logger.info(msg)

    if match_hist_and_fut:
        logger.info("Dispatching historical dataset to future experiments.")
        catalog = _dispatch_historical_to_future(catalog, id_columns)

    # Cut entries that do not match search criteria
    if exclusions:
        for k in exclusions.keys():
            ex = catalog.search(**{k: exclusions[k]})
            catalog.esmcat._df = pd.concat([catalog.df, ex.df]).drop_duplicates(keep=False)
            msg = f"Removing {len(ex.df)} assets based on exclusion dict '{k}': {exclusions[k]}."
            logger.info(msg)
    full_catalog = deepcopy(catalog)  # Used for searching for fixed fields
    if other_search_criteria:
        catalog = catalog.search(**other_search_criteria)
        msg = f"{len(catalog.df)} assets matched the criteria : {other_search_criteria}."
        logger.info(msg)
    if restrict_warming_level:
        if isinstance(restrict_warming_level, bool):
            restrict_warming_level = {}
        restrict_warming_level.setdefault("ignore_member", False)
        catalog.esmcat._df = _restrict_wl(catalog.df, restrict_warming_level)

    if id_columns is not None or catalog.df["id"].isnull().any():
        ids = generate_id(catalog.df, id_columns)
        if id_columns is not None:
            # Recreate id from user specifications
            catalog.df["id"] = ids
        else:
            # Only fill in the missing IDs.
            # Unreachable line if 'id' is in the aggregation control columns, but this is a safety measure.
            catalog.df["id"] = catalog.df["id"].fillna(ids)

    if catalog.df.empty:
        warnings.warn(
            "Found no match corresponding to the search criteria.",
            UserWarning,
            stacklevel=1,
        )
        return {}

    coverage_kwargs = coverage_kwargs or {}
    periods = standardize_periods(periods, out_dtype="datetime")

    msg = f"Iterating over {len(catalog.unique('id'))} potential datasets."
    logger.info(msg)
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
                                if ((i in scat.df.columns) and (not pd.isnull(scat.df[i].iloc[0])))
                            }
                            scat_id.pop("experiment", None)
                            scat_id.pop("member", None)
                            varcat = full_catalog.search(
                                **scat_id,
                                xrfreq=xrfreq,
                                variable=var_id,
                                require_all_on=["id", "xrfreq"],
                            )
                            if len(varcat) > 1:
                                varcat.esmcat._df = varcat.df.iloc[[0]]
                            if len(varcat) == 1:
                                warnings.warn(
                                    f"Dataset {sim_id} doesn't have the fixed field {var_id}, but it can be acquired from {varcat.df['id'].iloc[0]}.",
                                    UserWarning,
                                    stacklevel=1,
                                )
                                for i in {"member", "experiment", "id"}.intersection(varcat.df.columns):
                                    varcat.df.loc[:, i] = scat.df[i].iloc[0]

                        # TODO: Temporary fix until this is changed in intake_esm
                        varcat._requested_variables_true = [var_id]
                        varcat._dependent_variables = list(set(varcat._requested_variables).difference(varcat._requested_variables_true))
                    else:
                        # TODO: Add support for DerivedVariables that themselves require DerivedVariables
                        # TODO: Add support for DerivedVariables that exist on different frequencies (e.g. 1hr 'pr' & 3hr 'tas')
                        varcat = scat.search(variable=var_id, require_all_on=["id", "xrfreq"])
                        msg = f"At var {var_id}, after search cat has {varcat.derivedcat.keys()}"
                        logger.debug(msg)
                        # TODO: Temporary fix until this is changed in intake_esm
                        varcat._requested_variables_true = [var_id]
                        varcat._dependent_variables = list(set(varcat._requested_variables).difference(varcat._requested_variables_true))

                        # We want to match lines with the correct freq,
                        # IF allow_resampling is True and xrfreq translates to a timedelta,
                        # we also want those with (stricyly) higher temporal resolution
                        same_frq = varcat.df.xrfreq == xrfreq
                        td = pd.to_timedelta(CV.xrfreq_to_timedelta(xrfreq))
                        varcat.df["timedelta"] = pd.to_timedelta(varcat.df.xrfreq.apply(CV.xrfreq_to_timedelta, default="NAN"))
                        # else is joker (any timedelta)
                        lower_frq = np.less(varcat.df.timedelta, td) if pd.notnull(td) else False
                        varcat.esmcat._df = varcat.df[same_frq | (lower_frq & allow_resampling)]

                        # For each dataset (id * xrfreq * processing_level * domain * variable),
                        # make sure that file availability covers the requested time periods
                        if periods is not None and len(varcat) > 0:
                            valid_tp = []
                            for _var, group in varcat.df.groupby(
                                varcat.esmcat.aggregation_control.groupby_attrs + ["variable"],
                                observed=True,
                            ):
                                valid_tp.append(
                                    subset_file_coverage(group, periods, **coverage_kwargs)
                                )  # If valid, this returns the subset of files that cover the time period
                            varcat.esmcat._df = pd.concat(valid_tp)

                        # We now select the coarsest timedelta for each variable
                        # We need to re-iterate over variables in case we used the registry
                        # (and thus there are multiple variables in varcat)
                        rows = []
                        for _var, group in varcat.df.groupby("variable"):
                            rows.append(group[group.timedelta == group.timedelta.max()])
                        if rows:
                            # check if the requested variable exists and if so, remove DeriveVariable references
                            v_list = [rows[i]["variable"].iloc[0] for i in range(len(rows))]
                            v_list_check = [var_id in v_list[i] for i in range(len(v_list))]  # necessary in case a file has multiple variables
                            if any(v_list_check):
                                rows = [rows[v_list_check.index(True)]]
                                varcat.derivedcat = DerivedVariableRegistry()
                            varcat.esmcat._df = pd.concat(rows, ignore_index=True)
                        else:
                            varcat.esmcat._df = pd.DataFrame()

                    if varcat.df.empty:
                        msg = f"Dataset {sim_id} doesn't have all needed variables (missing at least {var_id})."
                        logger.debug(msg)
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
        msg = f"Found {len(catalogs)} with all variables requested and corresponding to the criteria."
        logger.info(msg)
    else:
        logger.warning("Found no match corresponding to the search criteria.")

    if restrict_resolution is not None and len(catalogs) > 0:
        catalogs = _restrict_by_resolution(catalogs, restrict_resolution, id_columns)

    if restrict_members is not None and len(catalogs) > 0:
        catalogs = _restrict_multimembers(catalogs, restrict_members, id_columns)

    return catalogs


@parse_config
def get_warming_level(*args, **kwargs) -> xr.Dataset | xr.DataArray | dict | pd.Series | pd.DataFrame | str | list:
    """
    Deprecated. Use get_period_from_warming_level instead.

    Parameters
    ----------
    args: list
        Arguments to pass to get_period_from_warming_level
    kwargs: dict
        Keyword arguments to pass to get_period_from_warming_level

    Returns
    -------
    xr.Dataset or xr.DataArray or dict or list or str
        Output of get_period_from_warming_level

    """
    kwargs = kwargs.copy()
    kwargs["return_central_year"] = not kwargs.get("return_horizon", True)
    kwargs.pop("return_horizon", None)
    warnings.warn(
        "get_warming_level has been deprecated. Use get_period_from_warming_level instead.",
        FutureWarning,
        stacklevel=2,
    )
    return get_period_from_warming_level(*args, **kwargs)


@parse_config
def get_period_from_warming_level(  # noqa: C901
    realization: (xr.Dataset | xr.DataArray | dict | pd.Series | pd.DataFrame | str | list),
    wl: float,
    *,
    window: int = 20,
    tas_baseline_period: Sequence[str] | None = None,
    ignore_member: bool = False,
    tas_src: str | os.PathLike | None = None,
    return_central_year: bool = False,
) -> xr.Dataset | xr.DataArray | dict | pd.Series | pd.DataFrame | str | list:
    """
    Use the IPCC Atlas method to return the window of time
    over which the requested level of global warming is first reached.

    Parameters
    ----------
    realization : xr.Dataset, xr.DataArray, dict, str, Series or sequence of those
       Model to be evaluated. Needs the four fields mip_era, source, experiment and member,
       as a dict or in a Dataset's attributes.
       Strings should follow this formatting: {mip_era}_{source}_{experiment}_{member}.
       Lists of dicts, strings or Datasets are also accepted, in which case the output will be a dict.
       Regex wildcards (.*) are accepted, but may lead to unexpected results.
       Datasets should include the catalogue attributes (starting by "cat:") required to create such a string:
       'cat:mip_era', 'cat:experiment', 'cat:member',
       and either 'cat:source' for global models or 'cat:driving_model' for regional models.
       e.g. 'CMIP5_CanESM2_rcp85_r1i1p1'
    wl : float, np.ndarray
       Warming level(s).
       e.g. 2 for a global warming level of +2 degree Celsius above the mean temperature of the `tas_baseline_period`.
    window : int
       Size of the rolling window in years over which to compute the warming level.
    tas_baseline_period : list, optional
       [start, end] of the base period. The warming is calculated with respect to it. The default is ["1850", "1900"].
    ignore_member : bool
       Decides whether to ignore the member when searching for the model run in tas_csv.
    tas_src : str, optional
       Path to a netCDF of annual global mean temperature (tas) with an annual "time" dimension
       and a "simulation" dimension with the following coordinates: "mip_era", "source", "experiment" and "member".
       If None, it will default to data/IPCC_annual_global_tas.nc which was built from
       the IPCC atlas data from  Iturbide et al., 2020 (https://doi.org/10.5194/essd-12-2959-2020)
       and extra data for missing CMIP6 models and pilot models of CRCM5 and ClimEx.
    return_central_year: bool
        If True, the output will be a string representing the middle of the period, using IPCC conventions in the case of an even window (y-9, y+10).
        If False (default), the output will be a list following the format ['start_yr', 'end_yr']

    Returns
    -------
    pd.Series or xr.DataArray or dict or list or str
        If `realization` is not a sequence, the output will follow the format indicated by `return_central_year`.
        If `realization` is a sequence, the output will be of the same type,
        with values following the format indicated by `return_central_year`.
    """
    tas_src = tas_src or Path(__file__).parent / "data" / "IPCC_annual_global_tas.nc"
    tas_baseline_period = standardize_periods(tas_baseline_period or ["1850", "1900"], multiple=False)

    if (window % 2) not in {0, 1}:
        raise ValueError(f"window should be an integer, received {type(window)}")

    FIELDS = ["mip_era", "source", "experiment", "member"]
    info_models = _wl_prep_infomodels(realization, ignore_member, FIELDS)

    # open nc
    tas = xr.open_dataset(tas_src).tas
    if np.isscalar(wl):
        wl = np.array([wl])

    def _get_warming_level(model):
        tas_sel = _wl_find_column(tas, model)
        if tas_sel is None:
            return None if return_central_year else [None, None]

        selected = "_".join([tas_sel[c].item() for c in FIELDS])
        msg = f"Computing warming level +{wl}°C for {model} from simulation: {selected}."
        logger.debug(msg)

        # compute reference temperature for the warming and difference from reference
        yearly_diff = tas_sel - tas_sel.sel(time=slice(*tas_baseline_period)).mean()

        # get the start and end date of the window when the warming level is first reached
        rolling_diff = yearly_diff.rolling(time=window, min_periods=window, center=True).mean()
        # shift(+1) is needed to reproduce IPCC results.
        # rolling defines the window as [n-10,n+9], but the the IPCC defines it as [n-9,n+10], where n is the center year.
        # interpolate will shift by -1, so +1 shift required on odd windows.

        if window % 2 != 0:  # odd window
            rolling_diff = rolling_diff.shift(time=1)
        # ensure series is monotonic -- keep only first year above point
        rolling_diff = rolling_diff.cumulative("time").max()
        # create interpolator
        interp = interp1d(
            rolling_diff,
            rolling_diff.time.dt.year,
            bounds_error=False,
            fill_value=(rolling_diff.time[0].dt.year.item(), np.nan),
            kind="previous",
            copy=False,
            assume_sorted=True,
        )
        # interpolate, make list,
        years = interp(wl).tolist()
        for i, year in enumerate(years):
            if np.isnan(year):
                years[i] = None if return_central_year else [None, None]
            else:
                years[i] = str(int(year)) if return_central_year else [str(int(year - window / 2 + 1)), str(int(year + window / 2))]
        if len(years) == 1:
            return years[0]
        return years

    out = list(map(_get_warming_level, info_models))
    if isinstance(realization, pd.DataFrame):
        index = realization.index
        if len(wl) > 1:
            index = [(i, w) for i in index for w in wl]
            out = [period for realization in out for period in realization]
        return pd.Series(out, index=index)
    if isinstance(realization, xr.DataArray):
        coords = {**realization.coords}
        dims = [realization.dims[0]]
        if len(wl) > 1:
            coords["wl"] = wl
            dims.append("wl")
        if return_central_year is False:
            coords["wl_bounds"] = [0, 1]
            dims.append("wl_bounds")
        return xr.DataArray(out, dims=dims, coords=coords)

    if len(out) == 1:
        return out[0]
    return out


@parse_config
def get_warming_level_from_period(
    realization: (xr.Dataset | xr.DataArray | dict | pd.Series | pd.DataFrame | str | list),
    period: list[str],
    *,
    tas_baseline_period: Sequence[str] | None = None,
    ignore_member: bool = False,
    tas_src: str | os.PathLike | None = None,
) -> xr.Dataset | xr.DataArray | dict | pd.Series | pd.DataFrame | float | list:
    """
    Return the warming level reached in the given period.

    Parameters
    ----------
    realization : xr.Dataset, xr.DataArray, dict, str, Series or sequence of those
       Model to be evaluated. Needs the four fields mip_era, source, experiment and member,
       as a dict or in a Dataset's attributes.
       Strings should follow this formatting: {mip_era}_{source}_{experiment}_{member}.
       Lists of dicts, strings or Datasets are also accepted, in which case the output will be a dict.
       Regex wildcards (.*) are accepted, but may lead to unexpected results.
       Datasets should include the catalogue attributes (starting by "cat:") required to create such a string:
       'cat:mip_era', 'cat:experiment', 'cat:member', and 'cat:source' for global models.
       For regional models : 'cat:mip_era', 'cat:experiment', 'cat:driving_member', and 'cat:driving_model'.
       e.g. 'CMIP5_CanESM2_rcp85_r1i1p1'
    period : list of str
       [start, end] of the period for which to compute the warming level.
    tas_baseline_period : list, optional
       [start, end] of the base period. The warming is calculated with respect to it. The default is ["1850", "1900"].
    ignore_member : bool
       Decides whether to ignore the member when searching for the model run in tas_csv.
    tas_src : str, optional
       Path to a netCDF of annual global mean temperature (tas) with an annual "time" dimension
       and a "simulation" dimension with the following coordinates: "mip_era", "source", "experiment" and "member".
       If None, it will default to data/IPCC_annual_global_tas.nc which was built from
       the IPCC atlas data from  Iturbide et al., 2020 (https://doi.org/10.5194/essd-12-2959-2020)
       and extra data for missing CMIP6 models and pilot models of CRCM5 and ClimEx.

    Returns
    -------
    pd.Series or xr.DataArray or float
        If `realization` is not a sequence, the output will be a float.
        If `realization` is a sequence, the output will be of the same type.
    """
    tas_src = tas_src or Path(__file__).parent / "data" / "IPCC_annual_global_tas.nc"
    tas_baseline_period = standardize_periods(tas_baseline_period or ["1850", "1900"], multiple=False)
    period = standardize_periods(period, multiple=False)

    FIELDS = ["mip_era", "source", "experiment", "member"]
    info_models = _wl_prep_infomodels(realization, ignore_member, FIELDS)

    # open nc
    tas = xr.open_dataset(tas_src).tas

    def _get_warming_level(model):
        tas_sel = _wl_find_column(tas, model)
        if tas_sel is None:
            return None

        selected = "_".join([tas_sel[c].item() for c in FIELDS])
        msg = f"Computing warming level during {period} for {model} from simulation: {selected}."
        logger.debug(msg)

        if not all(yr in tas_sel.time.dt.year for yr in np.arange(int(period[0]), int(period[1]) + 1)):
            raise ValueError(f"Period {period} is not fully covered by the provided 'tas_src' database for {selected}.")

        # compute reference temperature for the warming and difference from reference
        wl = tas_sel.sel(time=slice(*period)).mean() - tas_sel.sel(time=slice(*tas_baseline_period)).mean()

        return wl.item()

    out = list(map(_get_warming_level, info_models))
    if isinstance(realization, pd.DataFrame):
        return pd.Series(out, index=realization.index)
    if isinstance(realization, xr.DataArray):
        return xr.DataArray(out, dims=(realization.dims[0],), coords=realization.coords)

    if len(out) == 1:
        return out[0]
    return out


def _wl_prep_infomodels(realization, ignore_member, fields):
    if isinstance(realization, xr.Dataset | str | dict | pd.Series):
        reals = [realization]
    elif isinstance(realization, pd.DataFrame):
        reals = (row for i, row in realization.iterrows())
    elif isinstance(realization, xr.DataArray):
        reals = realization.values
    else:
        reals = realization

    info_models = []
    for real in reals:
        info = {}
        if isinstance(real, xr.Dataset):
            attrs = get_cat_attrs(real)
            # get info on ds
            if attrs.get("driving_model") is None:
                info["source"] = attrs["source"]
            else:
                info["source"] = attrs["driving_model"]
            info["experiment"] = attrs["experiment"]
            if ignore_member:
                info["member"] = ".*"
            elif attrs.get("driving_member") is None:
                info["member"] = attrs["member"]
            else:
                info["member"] = attrs["driving_member"]
            info["mip_era"] = attrs["mip_era"]
        elif isinstance(real, str):
            (
                info["mip_era"],
                info["source"],
                info["experiment"],
                info["member"],
            ) = real.split("_")
            if ignore_member:
                info["member"] = ".*"
        # Dict or Series (DataFrame row)
        elif hasattr(real, "keys") and set(real.keys()).issuperset((set(fields) - {"member"}) if ignore_member else fields):
            info = real
            if info.get("driving_model") is not None:
                info["source"] = info["driving_model"]
            if ignore_member:
                info["member"] = ".*"
            elif info.get("driving_member") is not None:
                info["member"] = info["driving_member"]
        else:
            raise ValueError(f"'realization' must be a Dataset, dict, string or list. Received {type(real)}.")
        info_models.append(info)

    return info_models


def _wl_find_column(tas, model):
    # Choose column based on ds cat attrs, +'$' to ensure a full match (matches end-of-string)
    mip = tas.mip_era.str.match(model["mip_era"] + "$")
    src = tas.source.str.match(model["source"] + "$")
    if not src.any():
        # Maybe it's an RCM, then requested source may contain the institute
        src = xr.apply_ufunc(model["source"].endswith, tas.source, vectorize=True)
    exp = tas.experiment.str.match(model["experiment"] + "$")
    mem = tas.member.str.match(model["member"] + "$")

    candidates = mip & src & exp & mem
    if not candidates.any():
        warnings.warn(f"No simulation fit the attributes of the input dataset ({model}).", stacklevel=2)
        return None

    if candidates.sum() > 1:
        logger.info("More than one simulation of the database fits the dataset metadata. Choosing the first one.")
    tas_sel = tas.isel(simulation=candidates.argmax("simulation"))
    return tas_sel


@parse_config
def subset_warming_level(
    ds: xr.Dataset,
    wl: float | Sequence[float],
    to_level: str = "warminglevel-{wl}vs{period0}-{period1}",
    wl_dim: str | bool = "+{wl}Cvs{period0}-{period1}",
    **kwargs,
) -> xr.Dataset | None:
    r"""
    Subsets the input dataset with only the window of time over which the requested level of global warming
    is first reached, using the IPCC Atlas method.
    A warming level is considered reached only if the full `window` years are available in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
       Input dataset.
       The dataset should include attributes to help recognize it and find its
       warming levels - 'cat:mip_era', 'cat:experiment', 'cat:member', and either
       'cat:source' for global models or 'cat:driving_institution' (optional) + 'cat:driving_model' for regional models.
       Or , it should include a `realization` dimension constructed as "{mip_era}_{source or driving_model}_{experiment}_{member}"
       for vectorized subsetting. Vectorized subsetting is currently only implemented for annual data.
    wl : float or sequence of floats
       Warming level.
       e.g. 2 for a global warming level of +2 degree Celsius above the mean temperature of the `tas_baseline_period`.
       Multiple levels can be passed, in which case using "{wl}" in  `to_level` and `wl_dim` is not recommended.
       Multiple levels are currently only implemented for annual data.
    to_level :
       The processing level to assign to the output.
       Use "{wl}", "{period0}" and "{period1}" in the string to dynamically include
       `wl`, 'tas_baseline_period[0]' and 'tas_baseline_period[1]'.
    wl_dim : str or boolean, optional
       The value to use to fill the new `warminglevel` dimension.
       Use "{wl}", "{period0}" and "{period1}" in the string to dynamically include
       `wl`, 'tas_baseline_period[0]' and 'tas_baseline_period[1]'.
       If None, no new dimensions will be added, invalid if `wl` is a sequence.
       If True, the dimension will include `wl` as numbers and units of "degC".
    \*\*kwargs :
        Instructions on how to search for warming levels, passed to :py:func:`get_period_from_warming_level`.

    Returns
    -------
    xr.Dataset or None
        Warming level dataset, or None if `ds` can't be subsetted for the requested warming level.
        The dataset will have a new dimension `warminglevel` with `wl_dim` as coordinates.
        If `wl` was a list or if ds had a "realization" dim, the "time" axis is replaced
        by a fake time starting in 1000-01-01 and with a length of `window` years.
        Start and end years of the subsets are bound in the new coordinate "warminglevel_bounds".
    """
    tas_baseline_period = standardize_periods(kwargs.get("tas_baseline_period", ["1850", "1900"]), multiple=False)
    window = kwargs.get("window", 20)

    # If wl was originally a list, this function is called a 2nd time with a generated fake_time
    fake_time = kwargs.pop("_fake_time", None)
    # Fake time generation is needed : real is a dim or multiple levels
    if fake_time is None and not isinstance(wl, int | float) or "realization" in ds.coords:
        freq = xr.infer_freq(ds.time)
        # FIXME: This is because I couldn't think of an elegant way to generate a fake_time otherwise.
        if not compare_offsets(freq, "==", "YS"):
            raise NotImplementedError(
                "Passing multiple warming levels or vectorizing subsetting along the 'realization' dim is currently not supported for non-annual data"
            )
        fake_time = xr.date_range("1000-01-01", periods=window, freq=freq, calendar=ds.time.dt.calendar)

    # If we got a wl sequence, call ourself multiple times and concatenate
    if not isinstance(wl, int | float):
        if not wl_dim or (isinstance(wl_dim, str) and "{wl}" not in wl_dim):
            raise ValueError("`wl_dim` must be True or a template string including '{wl}' if multiple levels are passed.")
        ds_wl = xr.concat(
            [
                subset_warming_level(
                    ds,
                    wli,
                    to_level=to_level,
                    wl_dim=wl_dim,
                    _fake_time=fake_time,
                    **kwargs,
                )
                for wli in wl
            ],
            "warminglevel",
        )
        return ds_wl

    # Creating the warminglevel coordinate
    if isinstance(wl_dim, str):  # a non-empty string
        wl_crd = xr.DataArray(
            [
                wl_dim.format(
                    wl=wl,
                    period0=tas_baseline_period[0],
                    period1=tas_baseline_period[1],
                )
            ],
            dims=("warminglevel",),
            name="warminglevel",
        )
    else:
        wl_crd = xr.DataArray([wl], dims=("warminglevel",), name="warminglevel", attrs={"units": "degC"})

    # For generating the bounds coord
    date_cls = xc.core.calendar.datetime_classes[ds.time.dt.calendar]
    if "realization" in ds.coords:
        # Vectorized subset
        realdim = ds.realization.dims[0]
        bounds = get_period_from_warming_level(ds.realization, wl, return_central_year=False, **kwargs)
        reals = []
        for real in bounds[realdim].values:
            start, end = bounds.sel({realdim: real}).values
            data = ds.sel({realdim: [real], "time": slice(start, end)})
            wl_not_reached = (start is None) or (data.time.size == 0) or ((data.time.dt.year[-1] - data.time.dt.year[0] + 1) != window)
            if not wl_not_reached:
                bnds_crd = [
                    date_cls(int(start), 1, 1),
                    date_cls(int(end) + 1, 1, 1) - datetime.timedelta(seconds=1),
                ]
            else:
                # In the case of not reaching the WL, data might be too short
                # We create it again with the proper length
                data = ds.sel({realdim: [real]}).isel(time=slice(0, fake_time.size)) * np.nan
                bnds_crd = [np.nan, np.nan]
            reals.append(
                data.expand_dims(warminglevel=wl_crd).assign_coords(
                    time=fake_time[: data.time.size],
                    warminglevel_bounds=(
                        (realdim, "warminglevel", "wl_bounds"),
                        [[bnds_crd]],
                    ),
                )
            )
        ds_wl = xr.concat(reals, realdim)
    else:
        # Scalar subset, single level
        start_yr, end_yr = get_period_from_warming_level(ds, wl=wl, return_central_year=False, **kwargs)
        # cut the window selected above and expand dims with wl_crd
        ds_wl = ds.sel(time=slice(start_yr, end_yr))
        wl_not_reached = (start_yr is None) or (ds_wl.time.size == 0) or ((ds_wl.time.dt.year[-1] - ds_wl.time.dt.year[0] + 1) != window)
        if fake_time is None:
            # WL not reached, not in ds, or not fully contained in ds.time
            if wl_not_reached:
                return None
            ds_wl = ds_wl.expand_dims(warminglevel=wl_crd)
        else:
            # WL not reached, not in ds, or not fully contained in ds.time
            if wl_not_reached:
                ds_wl = ds.isel(time=slice(0, fake_time.size)) * np.nan
                wlbnds = (("warminglevel", "wl_bounds"), [[np.nan, np.nan]])
            else:
                wlbnds = (
                    ("warminglevel", "wl_bounds"),
                    [
                        [
                            date_cls(int(start_yr), 1, 1),
                            date_cls(int(end_yr) + 1, 1, 1) - datetime.timedelta(seconds=1),
                        ]
                    ],
                )
            # We are in an iteration over multiple levels, put the fake time axis, but remember bounds
            ds_wl = ds_wl.expand_dims(warminglevel=wl_crd).assign_coords(
                time=fake_time[: ds_wl.time.size],
                warminglevel_bounds=wlbnds,
            )

    if to_level is not None:
        ds_wl.attrs["cat:processing_level"] = to_level.format(
            wl=wl,
            period0=tas_baseline_period[0],
            period1=tas_baseline_period[1],
        )

    if not wl_dim:
        ds_wl = ds_wl.squeeze("warminglevel", drop=True)
    else:
        ds_wl.warminglevel.attrs.update(
            baseline=f"{tas_baseline_period[0]}-{tas_baseline_period[1]}",
            long_name=f"Warming level for {window}-year periods since {tas_baseline_period[0]}-{tas_baseline_period[1]}",
        )
    return ds_wl


def _dispatch_historical_to_future(catalog: DataCatalog, id_columns: list[str] | None = None) -> DataCatalog:
    """
    Update a DataCatalog by recopying each "historical" entry to its corresponding future experiments.

    For examples, if an historical entry has corresponding "ssp245" and "ssp585" entries,
    then it is copied twice, with its "experiment" field modified accordingly.
    The original "historical" entry is removed. This way, a subsequent search of the catalog
    with "experiment='ssp245'" includes the _historical_ assets (with no apparent distinction).

    "Historical" assets that did not find a match are removed from the output catalog.

    Parameters
    ----------
    catalog : DataCatalog
        Catalog to be evaluated.
    id_columns : list of str, optional
        List of columns to be used to identify unique simulations.
        If None, defaults to ID_COLUMNS.

    Returns
    -------
    DataCatalog
        Catalog with the historical entries duplicated and modified to match the future experiments.
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
    df["same_hist_member"] = df[sim_id_no_exp].apply(lambda row: "_".join(row.values.astype(str)), axis=1)

    new_lines = []
    for group in df.same_hist_member.unique():
        sdf = df[df.same_hist_member == group]
        hist = sdf[sdf.experiment == "historical"]
        if hist.empty:
            continue
        if pd.isna(sdf.activity).any():
            warnings.warn(
                f"np.NaN was found in the activity column of {group}. The rows with np.NaN activity will be skipped."
                "If you want them to be included in the historical and future matching, "
                "please put a valid activity (https://xscen.readthedocs.io/en/latest/columns.html)."
                "For example, xscen expects experiment `historical` to have `CMIP` activity "
                "and experiments `sspXYZ` to have `ScenarioMIP` activity. ",
                stacklevel=2,
            )
        for activity_id in set(sdf.activity) - {"HighResMip", np.nan}:
            sub_sdf = sdf[sdf.activity == activity_id]
            for exp_id in set(sub_sdf.experiment) - {"historical", "piControl", np.nan}:
                exp_hist = hist.copy()
                exp_hist["experiment"] = exp_id
                exp_hist["activity"] = activity_id
                exp_hist["same_hist_member"] = group
                sim_ids = sub_sdf[sub_sdf.experiment == exp_id].id.unique()
                if len(sim_ids) > 1:
                    raise ValueError(f"Got multiple dataset ids where we expected only one... : {sim_ids}")
                exp_hist["id"] = sim_ids[0]

                # Remove fixed fields that already exist in the future experiment
                dupes = pd.concat(
                    [
                        exp_hist.loc[exp_hist["frequency"] == "fx"],
                        sub_sdf.loc[(sub_sdf["frequency"] == "fx") & (sub_sdf["experiment"] == exp_id)],
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

    df = pd.concat([df] + new_lines, ignore_index=True).drop(columns=["same_hist_member"])
    df = df[df.experiment != "historical"].reset_index(drop=True)
    return DataCatalog(
        {"esmcat": catalog.esmcat.model_dump(), "df": df},
        registry=catalog.derivedcat,
        drop_duplicates=False,
    )


def _restrict_by_resolution(catalogs: dict, restrictions: str, id_columns: list[str] | None = None) -> dict:
    """
    Update the results from search_data_catalogs by removing simulations with multiple resolutions available.

    Parameters
    ----------
    catalogs : dict
        Dictionary of DataCatalogs to be evaluated.
    restrictions : str
        Either 'finest' or 'coarsest'.
    id_columns : list of str, optional
        List of columns to be used to identify unique simulations.
        If None, defaults to ID_COLUMNS.

    Returns
    -------
    dict
        Catalogs with duplicate simulations removed according to the resolution restrictions.

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
    df["id_nodom"] = df[list(set(id_columns or ID_COLUMNS).intersection(df.columns).difference(["domain"]))].apply(
        lambda row: "_".join(map(str, filter(pd.notna, row.values))), axis=1
    )
    for i in pd.unique(df["id_nodom"]):
        df_sim = df[df["id_nodom"] == i]
        domains = pd.unique(df_sim["domain"])

        if len(domains) > 1:
            msg = f"Dataset {i} appears to have multiple resolutions."
            logger.info(msg)

            # For CMIP, the order is dictated by a list of grid labels
            if "MIP" in pd.unique(df_sim["activity"])[0]:
                order = np.array([])
                for d in domains:
                    match = [CV.infer_resolution("CMIP").index(r) for r in CV.infer_resolution("CMIP") if re.match(pattern=r, string=d)]
                    if len(match) != 1:
                        raise ValueError(f"'{d}' matches no known CMIP domain.")
                    order = np.append(order, match[0])

                if restrictions == "finest":
                    chosen = [np.sort(np.array(domains)[np.where(order == np.array(order).min())[0]])[0]]
                elif restrictions == "coarsest":
                    chosen = [np.sort(np.array(domains)[np.where(order == np.array(order).max())[0]])[-1]]
                else:
                    raise ValueError("'restrict_resolution' should be 'finest' or 'coarsest'")

            # Note: For CORDEX, the order is dictated by both the grid label
            # and the resolution itself as well as the domain name
            elif pd.unique(df_sim["activity"])[0] == "CORDEX":
                # Unique CORDEX domains
                cordex_doms = list(pd.unique(pd.Series([d.split("-")[0] for d in domains])))
                chosen = []

                for d in cordex_doms:
                    sub = [doms for doms in domains if d in doms]
                    order = [float(re.split(r"^([A-Z]{3})-([0-9\.]*)([i]{0,1})$", s)[2]) for s in sub]

                    if restrictions == "finest":
                        chosen.extend([np.sort(np.array(sub)[np.where(order == np.array(order).min())[0]])[0]])
                    elif restrictions == "coarsest":
                        chosen.extend([np.sort(np.array(sub)[np.where(order == np.array(order).max())[0]])[0]])
                    else:
                        raise ValueError("'restrict_resolution' should be 'finest' or 'coarsest'")

            else:
                msg = f"Dataset {i} seems to have multiple resolutions, but its activity is not yet recognized or supported."
                logger.warning(msg)
                chosen = list(domains)
                pass

            to_remove = pd.unique(df_sim[df_sim["domain"].isin(list(set(pd.unique(df_sim["domain"])).difference(chosen)))]["id"])

            for k in to_remove:
                msg = f"Removing {k} from the results."
                logger.info(msg)
                catalogs.pop(k)

    return catalogs


def _restrict_multimembers(catalogs: dict, restrictions: dict, id_columns: list[str] | None = None):
    """
    Update the results from search_data_catalogs by removing simulations with multiple members available.

    Uses regex to try and adequately detect and order the member's identification number, but only tested for 'r-i-p'.

    Parameters
    ----------
    catalogs : dict
        Dictionary of DataCatalogs to be evaluated.
    restrictions : dict
        Dictionary of restrictions to be applied. Currently only supports {'ordered': int}.
    id_columns : list of str, optional
        List of columns to be used to identify unique simulations.
        If None, defaults to ID_COLUMNS.

    Returns
    -------
    dict
        Catalogs where simulations with multiple members have been restricted to the requested maximum number.
    """
    df = pd.concat([catalogs[s].df for s in catalogs.keys()])
    # remove the member from the group_by
    df["id_nomem"] = df[list(set(id_columns or ID_COLUMNS).intersection(df.columns).difference(["member", "driving_member"]))].apply(
        lambda row: "_".join(map(str, filter(pd.notna, row.values))), axis=1
    )

    for i in pd.unique(df["id_nomem"]):
        df_sim = df[df["id_nomem"] == i]
        if "driving_member" in df_sim:
            # We create a compound member for sorting
            # We can't use fillna("") as the columns might be categorical.
            # This order (member + driving_member) makes it so that the driving member priority over the rcm member.
            fullmembers = df_sim.member.astype(str).replace("nan", "r1") + df_sim.driving_member.astype(str).replace("nan", "r1")
        else:
            fullmembers = df_sim.member.astype(str).replace("nan", "r1")
        members = pd.unique(fullmembers)

        if len(members) > 1:
            msg = f"Dataset {i} has {len(members)} valid members. Restricting as per requested."
            logger.info(msg)

            if "ordered" in restrictions:
                to_keep = natural_sort(members)[0 : restrictions["ordered"]]
            else:
                raise NotImplementedError("Subsetting multiple members currently only supports 'ordered'.")

            to_remove = pd.unique(df_sim[fullmembers.isin(list(set(members).difference(to_keep)))]["id"])

            for k in to_remove:
                msg = f"Removing {k} from the results."
                logger.info(msg)
                catalogs.pop(k)

    return catalogs


def _restrict_wl(df: pd.DataFrame, restrictions: dict):
    """
    Update the results from search_data_catalogs according to warming level restrictions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be evaluated.
    restrictions : dict
        Dictionary of restrictions to be applied. Entries are passed to get_period_from_warming_level.
        If 'wl' is present, the warming level csv will be used to remove simulations that do not reach the requested warming level.
        Otherwise, the warming level csv will be used to remove simulations that are not available in it.

    Returns
    -------
    df :
        Updated DataFrame.
    """
    restrictions.setdefault("wl", 0)
    to_keep = get_period_from_warming_level(df, return_central_year=True, **restrictions).notnull()
    removed = pd.unique(df[~to_keep]["id"])
    df = df[to_keep]
    msg = f"Removing the following datasets because of the restriction for warming levels: {list(removed)}"
    logger.info(msg)
    return df
