"""Ensemble statistics and weights."""

import inspect
import logging
import os
import warnings
from copy import deepcopy
from itertools import chain, groupby
from pathlib import Path

import numpy as np
import xarray as xr
from xclim import ensembles

from .catalog import DataCatalog
from .catutils import generate_id
from .config import parse_config
from .regrid import regrid_dataset
from .spatial import get_grid_mapping, subset
from .utils import clean_up, get_cat_attrs


logger = logging.getLogger(__name__)

__all__ = [
    "build_partition_data",
    "ensemble_stats",
    "generate_weights",
    "reduce_ensemble",
]


@parse_config
def ensemble_stats(  # noqa: C901
    datasets: (dict | list[str | os.PathLike] | list[xr.Dataset] | list[xr.DataArray] | xr.Dataset),
    statistics: dict,
    *,
    create_kwargs: dict | None = None,
    weights: xr.DataArray | None = None,
    common_attrs_only: bool = True,
    to_level: str = "ensemble",
) -> xr.Dataset:
    """
    Create an ensemble and computes statistics on it.

    Parameters
    ----------
    datasets : dict or list of [str, os.PathLike, Dataset or DataArray], or Dataset
        List of file paths or xarray Dataset/DataArray objects to include in the ensemble.
        A dictionary can be passed instead of a list, in which case the keys are used as coordinates along the new
        `realization` axis.
        Tip: With a project catalog, you can do: `datasets = pcat.search(**search_dict).to_dataset_dict()`.
        If a single Dataset is passed, it is assumed to already be an ensemble and will be used as is. The 'realization' dimension is required.
    statistics : dict
        xclim.ensembles statistics to be called. Dictionary in the format {function: arguments}.
        If a function requires 'weights', you can leave it out of this dictionary and
        it will be applied automatically if the 'weights' argument is provided.
        See the Notes section for more details on robustness statistics, which are more complex in their usage.
    create_kwargs : dict, optional
        Dictionary of arguments for xclim.ensembles.create_ensemble.
    weights : xr.DataArray, optional
        Weights to apply along the 'realization' dimension. This array cannot contain missing values.
    common_attrs_only : bool
        If True, keeps only the global attributes that are the same for all datasets and generate new id.
        If False, keeps global attrs of the first dataset (same behaviour as xclim.ensembles.create_ensemble)
    to_level : str
        The processing level to assign to the output.

    Returns
    -------
    xr.Dataset
        Dataset with ensemble statistics

    Notes
    -----
    * The positive fraction in 'change_significance' and 'robustness_fractions' is calculated by
      xclim using 'v > 0', which is not appropriate for relative deltas.
      This function will attempt to detect relative deltas by using the 'delta_kind' attribute ('rel.', 'relative', '*', or '/')
      and will apply 'v - 1' before calling the function.
    * The 'robustness_categories' statistic requires the outputs of 'robustness_fractions'.
      Thus, there are two ways to build the 'statistics' dictionary:

      1. Having 'robustness_fractions' and 'robustness_categories' as separate entries in the dictionary.
         In this case, all outputs will be returned.
      2. Having 'robustness_fractions' as a nested dictionary under 'robustness_categories'.
         In this case, only the robustness categories will be returned.

    * A 'ref' DataArray can be passed to 'change_significance' and 'robustness_fractions', which will be used by xclim to compute deltas
      and perform some significance tests. However, this supposes that both 'datasets' and 'ref' are still timeseries (e.g. annual means),
      not climatologies where the 'time' dimension represents the period over which the climatology was computed. Thus,
      using 'ref' is only accepted if 'robustness_fractions' (or 'robustness_categories') is the only statistic being computed.
    * If you want to use compute a robustness statistic on a climatology, you should first compute the climatologies and deltas yourself,
      then leave 'ref' as None and pass the deltas as the 'datasets' argument. This will be compatible with other statistics.

    See Also
    --------
    xclim.ensembles._base.create_ensemble, xclim.ensembles._base.ensemble_percentiles,
    xclim.ensembles._base.ensemble_mean_std_max_min,
    xclim.ensembles._robustness.robustness_fractions, xclim.ensembles._robustness.robustness_categories,
    xclim.ensembles._robustness.robustness_coefficient,
    """
    create_kwargs = create_kwargs or {}
    statistics = deepcopy(statistics)  # to avoid modifying the original dictionary

    # if input files are .zarr, change the engine automatically
    if isinstance(datasets, list) and isinstance(datasets[0], str | os.PathLike):
        path = Path(datasets[0])
        if path.suffix in [".zarr", ".zip"]:
            create_kwargs.setdefault("engine", "zarr")

    if not isinstance(datasets, xr.Dataset):
        ens = ensembles.create_ensemble(datasets, **create_kwargs)
    else:
        ens = datasets

    ens_stats = xr.Dataset(attrs=ens.attrs)

    # "robustness_categories" requires "robustness_fractions", but we want to compute things only once if both are requested.
    statistics_to_compute = list(statistics.keys())
    if "robustness_categories" in statistics_to_compute:
        if "robustness_fractions" in statistics_to_compute:
            statistics_to_compute.remove("robustness_fractions")
        elif "robustness_fractions" not in statistics["robustness_categories"]:
            raise ValueError(
                "'robustness_categories' requires 'robustness_fractions' to be computed. "
                "Either add 'robustness_fractions' to the statistics dictionary or "
                "add 'robustness_fractions' under the 'robustness_categories' dictionary."
            )

    for stat in statistics_to_compute:
        stats_kwargs = deepcopy(statistics.get(stat) or {})
        msg = f"Calculating {stat} from an ensemble of {len(ens.realization)} simulations."
        logger.info(msg)

        # Workaround for robustness_categories
        real_stat = None
        categories_kwargs = {}
        if stat == "robustness_categories":
            real_stat = "robustness_categories"
            stat = "robustness_fractions"
            categories_kwargs = deepcopy(stats_kwargs)
            categories_kwargs.pop("robustness_fractions", None)
            stats_kwargs = deepcopy(stats_kwargs.get("robustness_fractions", None) or statistics.get("robustness_fractions", {}))

        if weights is not None:
            if "weights" in inspect.getfullargspec(getattr(ensembles, stat))[0]:
                stats_kwargs["weights"] = weights.reindex_like(ens.realization)
            else:
                warnings.warn(f"Weighting is not supported for '{stat}'. The results may be incorrect.", stacklevel=2)

        if stat in [
            "robustness_fractions",
            "robustness_categories",
        ]:
            # These statistics only work on DataArrays
            for v in ens.data_vars:
                with xr.set_options(keep_attrs=True):
                    # Support for relative deltas [0, inf], where positive fraction is 'v > 1' instead of 'v > 0'.
                    delta_kind = ens[v].attrs.get("delta_kind")
                    if stats_kwargs.get("ref") is not None and delta_kind is not None:
                        raise ValueError(f"{v} is a delta, but 'ref' was still specified.")
                    if delta_kind in ["rel.", "relative", "*", "/"]:
                        msg = f"Relative delta detected for {v}. Applying 'v - 1' before change_significance."
                        logging.info(msg)
                        ens_v = ens[v] - 1
                    else:
                        ens_v = ens[v]

                    # Call the function
                    tmp = getattr(ensembles, stat)(ens_v, **stats_kwargs)

                    # Robustness categories
                    if real_stat == "robustness_categories":
                        categories = ensembles.robustness_categories(tmp, **categories_kwargs)
                        ens_stats[f"{v}_robustness_categories"] = categories

                    # Only return the robustness fractions if they were requested.
                    if "robustness_fractions" in statistics.keys():
                        tmp = tmp.rename({s: f"{v}_{s}" for s in tmp.data_vars})
                        ens_stats = ens_stats.merge(tmp)

        else:
            ens_stats = ens_stats.merge(getattr(ensembles, stat)(ens, **stats_kwargs))

    # delete the realization coordinate if there
    if "realization" in ens_stats:
        ens_stats = ens_stats.drop_vars("realization")

    # delete attrs that are not common to all dataset
    if common_attrs_only and not isinstance(datasets, xr.Dataset):
        # if they exist remove attrs specific to create_ensemble
        create_kwargs.pop("mf_flag", None)
        create_kwargs.pop("resample_freq", None)
        create_kwargs.pop("calendar", None)
        create_kwargs.pop("preprocess", None)

        ens_stats = clean_up(
            ds=ens_stats,
            common_attrs_only=datasets,
            common_attrs_open_kwargs=create_kwargs,
        )

    ens_stats.attrs["cat:processing_level"] = to_level
    ens_stats.attrs["ensemble_size"] = len(datasets)

    return ens_stats


def generate_weights(  # noqa: C901
    datasets: dict | list,
    *,
    independence_level: str = "model",
    balance_experiments: bool = False,
    attribute_weights: dict | None = None,
    skipna: bool = True,
    v_for_skipna: str | None = None,
    standardize: bool = False,
) -> xr.DataArray:
    """
    Use realization attributes to automatically generate weights along the 'realization' dimension.

    Parameters
    ----------
    datasets : dict
        List of Dataset objects that will be included in the ensemble.
        The datasets should include the necessary attributes to understand their metadata - See 'Notes' below.
        A dictionary can be passed instead of a list, in which case the keys are used for the 'realization' coordinate.
        Tip: With a project catalog, you can do: `datasets = pcat.search(**search_dict).to_dataset_dict()`.
    independence_level : str
        'model': Weights using the method '1 model - 1 Vote',
        where every unique combination of 'source' and 'driving_model' is considered a model.
        'GCM': Weights using the method '1 GCM - 1 Vote'
        'institution': Weights using the method '1 institution - 1 Vote'
    balance_experiments : bool
        If True, each experiment will be given a total weight of 1
        (prior to subsequent weighting made through `attribute_weights`).
        This option requires the 'cat:experiment' attribute to be present in all datasets.
    attribute_weights : dict, optional
        Nested dictionaries of weights to apply to each dataset.
        These weights are applied after the independence weighting.
        The first level of keys are the attributes for which weights are being given.
        The second level of keys are unique entries for the attribute, with the value being either an individual weight
        or a xr.DataArray. If a DataArray is used, its dimensions must be the same non-stationary coordinate
        as the datasets (ex: time, horizon) and the attribute being weighted (ex: experiment).
        A `others` key can be used to give the same weight to all entries not specifically named in the dictionary.
        Example #1: {'source': {'MPI-ESM-1-2-HAM': 0.25, 'MPI-ESM1-2-HR': 0.5}},
        Example #2: {'experiment': {'ssp585': xr.DataArray, 'ssp126': xr.DataArray}, 'institution': {'CCCma': 0.5, 'others': 1}}
    skipna : bool
        If True, weights will be computed from attributes only.
        If False, weights will be computed from the number of non-missing values.
        skipna=False requires either a 'time' or 'horizon' dimension in the datasets.
    v_for_skipna : str, optional
        Variable to use for skipna=False. If None, the first variable in the first dataset is used.
    standardize : bool
        If True, the weights are standardized to sum to 1 (per timestep/horizon, if skipna=False).

    Notes
    -----
    The following attributes are required for the function to work:
        - 'cat:source' in all datasets
        - 'cat:driving_model' in regional climate models
        - 'cat:institution' in all datasets if independence_level='institution'
        - 'cat:experiment' in all datasets if split_experiments=True

    Even when not required, the 'cat:member' and 'cat:experiment' attributes are strongly recommended to ensure the weights are computed correctly.

    Returns
    -------
    xr.DataArray
        Weights along the 'realization' dimension, or 2D weights along the 'realization' and 'time/horizon' dimensions if skipna=False.
    """
    if isinstance(datasets, list):
        datasets = {i: datasets[i] for i in range(len(datasets))}

    if independence_level not in ["model", "GCM", "institution"]:
        raise ValueError(f"'independence_level' should be between 'model', 'GCM', and 'institution', received {independence_level}.")
    if skipna is False:
        if v_for_skipna is None:
            v_for_skipna = list(datasets[list(datasets.keys())[0]].data_vars)[0]
            msg = f"Using '{v_for_skipna}' as the variable to check for missing values."
            logger.info(msg)

        # Check if any dataset has dimensions that are not 'time' or 'horizon'
        other_dims = {k: [d for d in datasets[k][v_for_skipna].dims if d not in ["time", "horizon"]] for k in datasets.keys()}
        for k in other_dims:
            if len(other_dims[k]) > 0:
                warnings.warn(
                    f"Dataset {k} has dimensions that are not 'time' or 'horizon': {other_dims[k]}. "
                    "The first indexes of these dimensions will be used to compute the weights.",
                    stacklevel=2,
                )
                datasets[k] = datasets[k].isel({d: 0 for d in other_dims[k]})

    # Use metadata to identify the simulation attributes
    keys = datasets.keys()
    defdict = {
        "experiment": None,
        "institution": None,
        "driving_model": None,
        "source": None,
        "member": None,
    }

    info = {key: dict(defdict, **get_cat_attrs(datasets[key])) for key in keys}

    # Check if there are both RCMs and GCMs in datasets, with attribute_weights set to weight them.
    if attribute_weights and (
        any(a in ["source", "driving_model"] for a in list(attribute_weights.keys()))
        and len(list(groupby([info[k]["driving_model"] is None for k in info.keys()]))) > 1
    ):
        raise NotImplementedError(
            "Weighting `source` and/or `driving_model` through `attribute_weights` is not yet implemented when given a mix of GCMs and RCMs."
        )

    # More easily manage GCMs and RCMs
    for k in info:
        if info[k]["driving_model"] is None or len(info[k]["driving_model"]) == 0:
            info[k]["driving_model"] = info[k]["source"]

    # Verifications
    if any((info[k]["driving_model"] is None or len(info[k]["driving_model"]) == 0) for k in info):
        raise ValueError("The 'cat:source' or 'cat:driving_model' attribute is missing from some simulations.")
    if balance_experiments and any((info[k]["experiment"] is None or len(info[k]["experiment"]) == 0) for k in info):
        raise ValueError("The 'cat:experiment' attribute is missing from some simulations. 'balance_experiments' cannot be True.")
    if independence_level == "institution" and any((info[k]["institution"] is None or len(info[k]["institution"]) == 0) for k in info):
        raise ValueError("The 'cat:institution' attribute is missing from some simulations. 'independence_level' cannot be 'institution'.")
    for attr in ["member", "experiment"]:
        if any(info[k][attr] is None for k in info):
            if all(info[k][attr] is None for k in info):
                warnings.warn(f"The 'cat:{attr}' attribute is missing from all datasets. Make sure the results are correct.", stacklevel=2)
            else:
                warnings.warn(f"The 'cat:{attr}' attribute is inconsistent across datasets. Results are likely to be incorrect.", stacklevel=2)

    # Combine the member and experiment attributes
    for k in info:
        info[k]["member-exp"] = str(info[k]["member"]) + "-" + str(info[k]["experiment"])

    # Build the weights according to the independence structure
    if skipna:
        weights = xr.DataArray(
            np.zeros(len(info.keys())),
            dims=["realization"],
            coords={"realization": list(info.keys())},
        )
    else:
        # Get the name of the extra dimension
        extra_dim = list(
            chain.from_iterable(
                [[datasets[list(keys)[d]][h] for h in ["time", "horizon"] if h in datasets[list(keys)[d]].dims] for d in range(len(keys))]
            )
        )
        if len({e.name for e in extra_dim}) != 1:
            raise ValueError(f"Expected either 'time' or 'horizon' as an extra dimension, found {extra_dim}.")

        # Combine the extra dimension and remove duplicates
        extra_dimension = xr.concat(extra_dim, dim=extra_dim[0].name).drop_duplicates(extra_dim[0].name)

        # Check that the extra dimension is the same for all datasets.
        # If not, modify the datasets to make them the same.
        if not all(extra_dimension.equals(extra_dim[d]) for d in range(len(extra_dim))):
            warnings.warn(f"Extra dimension {extra_dimension.name} is not the same for all datasets. Reindexing.", stacklevel=2)
            for d in datasets.keys():
                datasets[d] = datasets[d].reindex({extra_dimension.name: extra_dimension})

        weights = xr.DataArray(
            np.zeros((len(info.keys()), len(extra_dimension))),
            dims=["realization", extra_dimension.name],
            coords={
                "realization": list(info.keys()),
                extra_dimension.name: extra_dimension,
            },
        )

    for i in range(len(info)):
        sim = info[list(keys)[i]]

        # Number of models running a given realization of a driving model
        models_struct = ["source", "driving_model", "member-exp"] if independence_level == "model" else ["driving_model", "member-exp"]
        models = [k for k in info.keys() if all([info[k][s] == sim[s] for s in models_struct])]

        if skipna:
            n_models = len(models)
        else:
            n_models = xr.concat([datasets[k][v_for_skipna].notnull() for k in models], dim="realization").sum(dim="realization")

        # Number of realizations of a given driving model
        if independence_level == "model":
            realization_struct = ["source", "driving_model", "experiment"] if balance_experiments else ["source", "driving_model"]
        else:
            realization_struct = ["driving_model", "experiment"] if balance_experiments else ["driving_model"]
        realizations = {info[k]["member-exp"] for k in info.keys() if all([info[k][s] == sim[s] for s in realization_struct])}

        if skipna:
            n_realizations = len(realizations)
        else:
            n_realizations = xr.zeros_like(datasets[list(keys)[0]][v_for_skipna])
            r_models = dict()
            for r in realizations:
                r_models[r] = [k for k in info.keys() if (all([info[k][s] == sim[s] for s in realization_struct]) and (info[k]["member-exp"] == r))]
                n_realizations = n_realizations + (
                    xr.concat(
                        [datasets[k][v_for_skipna].notnull() for k in r_models[r]],
                        dim="realization",
                    ).sum(dim="realization")
                    > 0
                )

        # Number of driving models run by a given institution
        if independence_level == "institution":
            institution_struct = ["institution", "experiment"] if balance_experiments else ["institution"]
            institution = {info[k]["driving_model"] for k in info.keys() if all([info[k][s] == sim[s] for s in institution_struct])}

            if skipna:
                n_institutions = len(institution)
            else:
                n_institutions = xr.zeros_like(datasets[list(keys)[0]][v_for_skipna])
                i_models = dict()
                for ii in institution:
                    i_models[ii] = [
                        k for k in info.keys() if (all([info[k][s] == sim[s] for s in institution_struct]) and (info[k]["driving_model"] == ii))
                    ]
                    n_institutions = n_institutions + (
                        xr.concat(
                            [datasets[k][v_for_skipna].notnull() for k in i_models[ii]],
                            dim="realization",
                        ).sum(dim="realization")
                        > 0
                    )
        else:
            n_institutions = 1

        # Divide the weight equally between the group
        w = 1 / n_models / n_realizations / n_institutions
        weights[i] = xr.where(np.isfinite(w), w, 0)

    if balance_experiments:
        # Divide the weight equally between the experiments
        experiments = [info[k]["experiment"] for k in info.keys()]
        weights = weights.assign_coords(
            experiment=("realization", experiments),
        )
        expsum = weights.groupby("experiment").sum(dim="realization")

        for e in expsum.experiment:
            weights = weights.where(weights.experiment != e, other=weights / expsum.sel(experiment=e))

        # Drop the experiment coordinate
        weights = weights.drop_vars("experiment")

    # Attribute_weights
    if attribute_weights:
        stationary_weights = {}
        non_stationary_weights = {}
        for att, v_att in attribute_weights.items():
            # Add warning when a mismatch between independance_level/experiment_weight and attribute_weights is detected
            if att == "experiment" and not balance_experiments:
                warnings.warn("Key experiment given in attribute_weights without argument balance_experiments=True", stacklevel=2)

            if (
                (att == "source" and independence_level != "model")
                or (att == "driving_model" and independence_level != "GCM")
                or (att == "institution" and independence_level != "institution")
            ):
                warnings.warn(f"The {att} weights do not match the {independence_level} independence_level", stacklevel=2)

            # Verification
            if att not in info[k] or any((info[k][att] is None or len(info[k][att]) == 0) for k in info):
                raise ValueError(f"The {att} attribute is missing from some simulations.")
            # Split dict and xr.DataArray weights
            if isinstance(v_att, xr.DataArray):
                non_stationary_weights[att] = v_att
            elif isinstance(v_att, dict):
                stationary_weights[att] = v_att
            else:
                raise ValueError("Attribute_weights should be dict or xr.DataArray.")
        # Stationary weights (dicts)
        if stationary_weights:
            for att, w_dict in stationary_weights.items():
                for k, v in info.items():
                    if v[att] not in w_dict and "others" not in w_dict:
                        raise ValueError(f"The {att} {v[att]} or others are not in the attribute_weights dict.")
                    elif v[att] not in w_dict and "others" in w_dict:
                        w = w_dict["others"]
                    elif v[att] in w_dict:
                        w = w_dict[v[att]]
                    weights.loc[{"realization": k}] = weights.sel(realization=k) * w
        # Non-stationary weights (xr.DataArray)
        if non_stationary_weights:
            for att, da in non_stationary_weights.items():
                # check if the attribute is in the xr.DataArray coords
                if att not in da.coords:
                    raise ValueError(f"{att} is not in the xr.DataArray coords.")
                # find the coordinate (coord) to broadcast the weights (not equal to the attribute), ex: time / horizon
                ls_coord = list(da.coords)
                ls_coord.remove(att)
                if len(ls_coord) > 1:
                    raise ValueError(f"The {att} DataArray has more than one coord dimension to apply weights.")
                else:
                    coord = ls_coord[0]
                # broadcast coord to the weights DataArray
                if coord not in weights.coords:
                    weights = weights.expand_dims({coord: da[coord].values})
                ls_da = []
                for v in info.values():
                    if v[att] not in da[att] and "others" not in da[att]:
                        raise ValueError(f"The {att} {v[att]} or others are not in the attribute_weights datarray coords.")
                    elif v[att] not in da[att] and "others" in da[att]:
                        ls_da.append(da.sel(**{att: "others"}).drop_vars(att))
                    else:
                        ls_da.append(da.sel(**{att: v[att]}).drop_vars(att))
                nw = xr.concat(ls_da, weights.realization)
                weights = weights * nw

    if standardize:
        weights = weights / weights.sum(dim="realization")

    return weights


def _partition_from_list(datasets, partition_dim, subset_kw, regrid_kw):
    list_ds = []
    # only keep attrs common to all datasets
    common_attrs = False
    for ds in datasets:
        if subset_kw:
            ds = subset(ds, **subset_kw)
            gridmap = get_grid_mapping(ds)
            ds = ds.drop_vars(
                [
                    ds.cf["longitude"].name,
                    ds.cf["latitude"].name,
                    ds.cf.axes["X"][0],
                    ds.cf.axes["Y"][0],
                    gridmap,
                ],
                errors="ignore",
            )

        if regrid_kw:
            ds = regrid_dataset(ds, **regrid_kw)

        for dim in partition_dim:
            if f"cat:{dim}" in ds.attrs:
                ds = ds.expand_dims(**{dim: [ds.attrs[f"cat:{dim}"]]})

        if "bias_adjust_project" in ds.dims:
            ds = ds.assign_coords(
                adjustment=(
                    "bias_adjust_project",
                    [ds.attrs.get("cat:adjustment", np.nan)],
                )
            )
            ds = ds.assign_coords(
                reference=(
                    "bias_adjust_project",
                    [ds.attrs.get("cat:reference", np.nan)],
                )
            )

        if "realization" in partition_dim:
            new_source = f"{ds.attrs['cat:institution']}_{ds.attrs['cat:source']}_{ds.attrs['cat:member']}"
            ds = ds.expand_dims(realization=[new_source])

        a = ds.attrs
        a.pop("intake_esm_vars", None)  # remove list for intersection to work
        common_attrs = dict(common_attrs.items() & a.items()) if common_attrs else a
        list_ds.append(ds)
    ens = xr.merge(list_ds)
    ens.attrs = common_attrs
    return ens


def _partition_from_catalog(datasets, partition_dim, subset_kw, regrid_kw, to_dataset_kw):
    if ("adjustment" in partition_dim or "reference" in partition_dim) and ("bias_adjust_project" in partition_dim):
        raise ValueError("The partition_dim can have either adjustment and reference or bias_adjust_project, not both.")

    if ("realization" in partition_dim) and ("source" in partition_dim):
        raise ValueError("The partition_dim can have either realization or source, not both.")

    # special case to handle source (create one dimension with institution_source_member)
    ensemble_on_list = None
    if "realization" in partition_dim:
        partition_dim.remove("realization")
        ensemble_on_list = ["institution", "source", "member"]

    subcat = datasets

    # get attrs that are common to all datasets
    common_attrs = {}
    for col, series in subcat.df.items():
        if (series[0] == series).all():
            common_attrs[f"cat:{col}"] = series[0]

    col_id = [
        ("adjustment" if "adjustment" in partition_dim else None),  # instead of bias_adjust_project, need to use adjustment, not method bc .sel
        ("reference" if "reference" in partition_dim else None),  # instead of bias_adjust_project
        "bias_adjust_project" if "bias_adjust_project" in partition_dim else None,
        "mip_era",
        "activity",
        "driving_model",
        "institution" if "realization" in partition_dim else None,
        "source",
        "experiment",
        "member" if "realization" in partition_dim else None,
        "domain",
    ]

    subcat.df["id"] = generate_id(subcat.df, col_id)

    # create a dataset for each bias_adjust_project, modify grid and concat them
    # choose dim that exists in partition_dim and first in the order of preference
    order_of_preference = ["reference", "bias_adjust_project", "source"]
    dim_with_different_grid = list(set(partition_dim) & set(order_of_preference))[0]

    list_ds = []
    for d in subcat.df[dim_with_different_grid].unique():
        ds = subcat.search(**{dim_with_different_grid: d}).to_dataset(
            concat_on=partition_dim,
            create_ensemble_on=ensemble_on_list,
            **to_dataset_kw,
        )

        if subset_kw:
            ds = subset(ds, **subset_kw)
            gridmap = get_grid_mapping(ds)
            ds = ds.drop_vars(
                [
                    ds.cf["longitude"].name,
                    ds.cf["latitude"].name,
                    ds.cf.axes["X"][0],
                    ds.cf.axes["Y"][0],
                    gridmap,
                ],
                errors="ignore",
            )

        if regrid_kw:
            ds = regrid_dataset(ds, **regrid_kw)

        # add coords adjustment and reference
        if "bias_adjust_project" in ds.dims:
            ds = ds.assign_coords(
                adjustment=(
                    "bias_adjust_project",
                    [ds.attrs.get("cat:adjustment", np.nan)],
                )
            )  # need to use adjustment, not method bc .sel
            ds = ds.assign_coords(
                reference=(
                    "bias_adjust_project",
                    [ds.attrs.get("cat:reference", np.nan)],
                )
            )
        list_ds.append(ds)
    ens = xr.concat(list_ds, dim=dim_with_different_grid)
    ens.attrs = common_attrs
    return ens


def build_partition_data(
    datasets: dict | list[xr.Dataset],
    partition_dim: list[str] | None = None,
    subset_kw: dict | None = None,
    regrid_kw: dict | None = None,
    rename_dict: dict | None = None,
    to_dataset_kw: dict | None = None,
    to_level: str = "partition-ensemble",
):
    """
    Get the input for the xclim partition functions.

    From a list or dictionary of datasets, create a single dataset with
    `partition_dim` dimensions (and time) to pass to one of the xclim partition functions
    (https://xclim.readthedocs.io/en/stable/api.html#uncertainty-partitioning).
    If the inputs have different grids,
    they have to be subsetted and/or regridded to a common grid/point.

    Parameters
    ----------
    datasets : list[xr.Dataset], dict[str, xr.Dataset], DataCatalog
        Either a list/dictionary of Datasets or a DataCatalog that will be included in the ensemble.
        The datasets should include the necessary ("cat:") attributes to understand their metadata.
        Tip: A dictionary can be created with `datasets = pcat.search(**search_dict).to_dataset_dict()`.

        The use of a DataCatalog is recommended for large ensembles.
        In that case, the ensembles will be loaded separately for each `bias_adjust_project`,
        the subsetting or regridding can be applied before combining the datasets through concatenation.
        If `bias_adjust_project` is not in `partition_dim`, `source` will be used instead.
    partition_dim: list[str]
        Components of the partition. They will become the dimension of the output.
        The default is ['source', 'experiment', 'bias_adjust_project'].
        For source, the dimension will actually be institution_source_member.
    subset_kw : dict, optional
        Arguments to pass to `xs.spatial.subset()`.
    regrid_kw : dict, optional
        Arguments to pass to `xs.regrid_dataset()`.
        Note that regriding is computationally expensive. For large datasets,
        it might be worth it to do the regridding first, outside of this function.
    rename_dict : dict, optional
        Dictionary to rename the dimensions from xscen names to xclim names.
        The default is {'source': 'model', 'bias_adjust_project': 'downscaling', 'experiment': 'scenario'}.
    to_dataset_kw : dict, optional
        Arguments to pass to `xscen.DataCatalog.to_dataset()` if datasets is a DataCatalog.
    to_level: str
        The processing level of the output dataset. Default is 'partition-ensemble'.

    Returns
    -------
    xr.Dataset
        The input data for the partition functions.

    See Also
    --------
    xclim.ensembles
    """
    if partition_dim is None:
        partition_dim = ["realization", "experiment", "bias_adjust_project"]
    if isinstance(datasets, dict):
        datasets = list(datasets.values())
    # initialize dict
    subset_kw = subset_kw or {}
    regrid_kw = regrid_kw or {}
    to_dataset_kw = to_dataset_kw or {}

    if isinstance(datasets, list):
        ens = _partition_from_list(datasets, partition_dim, subset_kw, regrid_kw)

    elif isinstance(datasets, DataCatalog):
        ens = _partition_from_catalog(datasets, partition_dim, subset_kw, regrid_kw, to_dataset_kw)

    else:
        raise ValueError("'datasets' should be a list/dictionary of xarray datasets or a xscen.DataCatalog")

    rename_dict = rename_dict or {}
    rename_dict.setdefault("realization", "model")
    rename_dict.setdefault("source", "model")
    rename_dict.setdefault("experiment", "scenario")
    rename_dict.setdefault("bias_adjust_project", "downscaling")
    rename_dict = {k: v for k, v in rename_dict.items() if k in ens.dims}
    ens = ens.rename(rename_dict)

    ens.attrs["cat:processing_level"] = to_level
    ens.attrs["cat:id"] = generate_id(ens)[0]

    return ens


@parse_config
def reduce_ensemble(
    data: xr.DataArray | dict | list | xr.Dataset,
    method: str,
    *,
    horizons: list[str] | None = None,
    create_kwargs: dict | None = None,
    **kwargs,
):
    r"""
    Reduce an ensemble of simulations using clustering algorithms from xclim.ensembles.

    Parameters
    ----------
    data : xr.DataArray
        Selection criteria data : 2-D xr.DataArray with dimensions 'realization' and 'criteria'.
        These are the values used for clustering. Realizations represent the individual original
        ensemble members and criteria the variables/indicators used in the grouping algorithm.
        This data can be generated using py:func:`xclim.ensembles.make_criteria`.
        Alternatively, either a xr.Dataset, a list of xr.Dataset or a dictionary of xr.Dataset can be passed,
        in which case the data will be built using py:func:`xclim.ensembles.create_ensemble` and py:func:`xclim.ensembles.make_criteria`.
    method : str
      ['kkz', 'kmeans']. Clustering method.
    horizons : list of str, optional
        Subset of horizons on which to create the data. Only used if `data` needs to be built.
    create_kwargs : dict, optional
        Arguments to pass to py:func:`xclim.ensembles.create_ensemble` if `data` is not an xr.DataArray.
    \*\*kwargs : dict
        Arguments to send to either py:func:`xclim.ensembles.kkz_reduce_ensemble` or py:func:`xclim.ensembles.kmeans_reduce_ensemble`.

    Returns
    -------
    selected : xr.DataArray
        DataArray of dimension 'realization' with the selected simulations.
    clusters : dict
        If using kmeans clustering, realizations grouped by cluster.
    fig_data : dict
        If using kmeans clustering, data necessary to call py:func:`xclim.ensembles.plot_rsqprofile`.

    Notes
    -----
    If building `data` to be constructed by this function, the datasets should already have a climatology computed on them, such that the data
    has no temporal dimension aside from the "horizon" coordinate (which is optional and might be used to subset the data).
    If the indicators are a mix of yearly, seasonal, and monthly, they should be stacked on the same time/horizon axis and put in the same dataset.
    You can use py:func:`xscen.utils.unstack_dates` on seasonal or monthly indicators to this end.
    """
    if isinstance(data, list | dict):
        data = ensembles.create_ensemble(datasets=data, **(create_kwargs or {}))
    if horizons:
        if "horizon" not in data.dims:
            raise ValueError("Data must have a 'horizon' dimension to be subsetted.")
        data = data.sel(horizon=horizons)
    if "criteria" not in data.dims:
        data = ensembles.make_criteria(data)

    selected = getattr(ensembles, f"{method}_reduce_ensemble")(data=data, **kwargs)

    clusters = {}
    fig_data = {}
    if method == "kmeans":
        fig_data = selected[2]
        clusters_tmp = selected[1]
        selected = selected[0]
        realization = np.arange(len(clusters_tmp))

        clusters = {g: data.realization.isel(realization=realization[clusters_tmp == g]) for g in np.unique(clusters_tmp)}
    selected = data.realization.isel(realization=selected)

    return selected, clusters, fig_data
