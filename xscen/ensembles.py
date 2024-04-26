"""Ensemble statistics and weights."""

import inspect
import logging
import os
import warnings
from copy import deepcopy
from itertools import chain, groupby
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr
from xclim import ensembles

from .config import parse_config
from .indicators import compute_indicators
from .regrid import regrid_dataset
from .spatial import subset
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
    datasets: Union[
        dict,
        list[Union[str, os.PathLike]],
        list[xr.Dataset],
        list[xr.DataArray],
        xr.Dataset,
    ],
    statistics: dict,
    *,
    create_kwargs: Optional[dict] = None,
    weights: Optional[xr.DataArray] = None,
    common_attrs_only: bool = True,
    to_level: str = "ensemble",
) -> xr.Dataset:
    """Create an ensemble and computes statistics on it.

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
    if isinstance(datasets, list) and isinstance(datasets[0], (str, os.PathLike)):
        path = Path(datasets[0])
        if path.suffix == ".zarr":
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
        logger.info(
            f"Calculating {stat} from an ensemble of {len(ens.realization)} simulations."
        )

        # Workaround for robustness_categories
        real_stat = None
        if stat == "robustness_categories":
            real_stat = "robustness_categories"
            stat = "robustness_fractions"
            categories_kwargs = deepcopy(stats_kwargs)
            categories_kwargs.pop("robustness_fractions", None)
            stats_kwargs = deepcopy(
                stats_kwargs.get("robustness_fractions", None)
                or statistics.get("robustness_fractions", {})
            )

        if weights is not None:
            if "weights" in inspect.getfullargspec(getattr(ensembles, stat))[0]:
                stats_kwargs["weights"] = weights.reindex_like(ens.realization)
            else:
                warnings.warn(
                    f"Weighting is not supported for '{stat}'. The results may be incorrect."
                )

        # FIXME: change_significance is deprecated and will be removed in xclim 0.49.
        if stat in [
            "change_significance",
            "robustness_fractions",
            "robustness_categories",
        ]:
            # FIXME: This can be removed once change_significance is removed.
            #  It's here because the 'ref' default was removed for change_significance in xclim 0.47.
            stats_kwargs.setdefault("ref", None)
            if (stats_kwargs.get("ref") is not None) and len(statistics_to_compute) > 1:
                raise ValueError(
                    f"The input requirements for '{stat}' when 'ref' is specified are not compatible with other statistics."
                )

            # These statistics only work on DataArrays
            for v in ens.data_vars:
                with xr.set_options(keep_attrs=True):
                    # Support for relative deltas [0, inf], where positive fraction is 'v > 1' instead of 'v > 0'.
                    delta_kind = ens[v].attrs.get("delta_kind")
                    if stats_kwargs.get("ref") is not None and delta_kind is not None:
                        raise ValueError(
                            f"{v} is a delta, but 'ref' was still specified."
                        )
                    if delta_kind in ["rel.", "relative", "*", "/"]:
                        logging.info(
                            f"Relative delta detected for {v}. Applying 'v - 1' before change_significance."
                        )
                        ens_v = ens[v] - 1
                    else:
                        ens_v = ens[v]

                    # Call the function
                    tmp = getattr(ensembles, stat)(ens_v, **stats_kwargs)

                    # Manage the multiple outputs of change_significance
                    # FIXME: change_significance is deprecated and will be removed in xclim 0.49.
                    if (
                        stat == "change_significance"
                        and stats_kwargs.get("p_vals", False) is False
                    ):
                        ens_stats[f"{v}_change_frac"], ens_stats[f"{v}_pos_frac"] = tmp
                    elif stat == "change_significance" and stats_kwargs.get(
                        "p_vals", False
                    ):
                        (
                            ens_stats[f"{v}_change_frac"],
                            ens_stats[f"{v}_pos_frac"],
                            ens_stats[f"{v}_p_vals"],
                        ) = tmp

                    # Robustness categories
                    if real_stat == "robustness_categories":
                        categories = ensembles.robustness_categories(
                            tmp, **categories_kwargs
                        )
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
    datasets: Union[dict, list],
    *,
    independence_level: str = "model",
    balance_experiments: bool = False,
    attribute_weights: Optional[dict] = None,
    skipna: bool = True,
    v_for_skipna: Optional[str] = None,
    standardize: bool = False,
    experiment_weights: bool = False,
) -> xr.DataArray:
    """Use realization attributes to automatically generate weights along the 'realization' dimension.

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
    experiment_weights : bool
        Deprecated. Use balance_experiments instead.

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
    if experiment_weights is True:
        warnings.warn(
            "`experiment_weights` has been renamed and will be removed in a future release. Use `balance_experiments` instead.",
            category=FutureWarning,
        )
        balance_experiments = True

    if isinstance(datasets, list):
        datasets = {i: datasets[i] for i in range(len(datasets))}

    if independence_level == "all":
        warnings.warn(
            "The independence level 'all' is deprecated and will be removed in a future version. Use 'model' instead.",
            category=FutureWarning,
        )
        independence_level = "model"

    if independence_level not in ["model", "GCM", "institution"]:
        raise ValueError(
            f"'independence_level' should be between 'model', 'GCM', and 'institution', received {independence_level}."
        )
    if skipna is False:
        if v_for_skipna is None:
            v_for_skipna = list(datasets[list(datasets.keys())[0]].data_vars)[0]
            logger.info(
                f"Using '{v_for_skipna}' as the variable to check for missing values."
            )

        # Check if any dataset has dimensions that are not 'time' or 'horizon'
        other_dims = {
            k: [
                d
                for d in datasets[k][v_for_skipna].dims
                if d not in ["time", "horizon"]
            ]
            for k in datasets.keys()
        }
        for k in other_dims:
            if len(other_dims[k]) > 0:
                warnings.warn(
                    f"Dataset {k} has dimensions that are not 'time' or 'horizon': {other_dims[k]}. "
                    "The first indexes of these dimensions will be used to compute the weights."
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
        and len(list(groupby([info[k]["driving_model"] is None for k in info.keys()])))
        > 1
    ):
        raise NotImplementedError(
            "Weighting `source` and/or `driving_model` through `attribute_weights` "
            "is not yet implemented when given a mix of GCMs and RCMs."
        )

    # More easily manage GCMs and RCMs
    for k in info:
        if info[k]["driving_model"] is None or len(info[k]["driving_model"]) == 0:
            info[k]["driving_model"] = info[k]["source"]

    # Verifications
    if any(
        (info[k]["driving_model"] is None or len(info[k]["driving_model"]) == 0)
        for k in info
    ):
        raise ValueError(
            "The 'cat:source' or 'cat:driving_model' attribute is missing from some simulations."
        )
    if balance_experiments and any(
        (info[k]["experiment"] is None or len(info[k]["experiment"]) == 0) for k in info
    ):
        raise ValueError(
            "The 'cat:experiment' attribute is missing from some simulations. 'balance_experiments' cannot be True."
        )
    if independence_level == "institution" and any(
        (info[k]["institution"] is None or len(info[k]["institution"]) == 0)
        for k in info
    ):
        raise ValueError(
            "The 'cat:institution' attribute is missing from some simulations. 'independence_level' cannot be 'institution'."
        )
    for attr in ["member", "experiment"]:
        if any(info[k][attr] is None for k in info):
            if all(info[k][attr] is None for k in info):
                warnings.warn(
                    f"The 'cat:{attr}' attribute is missing from all datasets. Make sure the results are correct."
                )
            else:
                warnings.warn(
                    f"The 'cat:{attr}' attribute is inconsistent across datasets. Results are likely to be incorrect."
                )

    # Combine the member and experiment attributes
    for k in info:
        info[k]["member-exp"] = (
            str(info[k]["member"]) + "-" + str(info[k]["experiment"])
        )

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
                [
                    [
                        datasets[list(keys)[d]][h]
                        for h in ["time", "horizon"]
                        if h in datasets[list(keys)[d]].dims
                    ]
                    for d in range(len(keys))
                ]
            )
        )
        if len({e.name for e in extra_dim}) != 1:
            raise ValueError(
                f"Expected either 'time' or 'horizon' as an extra dimension, found {extra_dim}."
            )

        # Combine the extra dimension and remove duplicates
        extra_dimension = xr.concat(extra_dim, dim=extra_dim[0].name).drop_duplicates(
            extra_dim[0].name
        )

        # Check that the extra dimension is the same for all datasets.
        # If not, modify the datasets to make them the same.
        if not all(extra_dimension.equals(extra_dim[d]) for d in range(len(extra_dim))):
            warnings.warn(
                f"Extra dimension {extra_dimension.name} is not the same for all datasets. Reindexing."
            )
            for d in datasets.keys():
                datasets[d] = datasets[d].reindex(
                    {extra_dimension.name: extra_dimension}
                )

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
        models_struct = (
            ["source", "driving_model", "member-exp"]
            if independence_level == "model"
            else ["driving_model", "member-exp"]
        )
        models = [
            k for k in info.keys() if all([info[k][s] == sim[s] for s in models_struct])
        ]

        if skipna:
            n_models = len(models)
        else:
            n_models = xr.concat(
                [datasets[k][v_for_skipna].notnull() for k in models], dim="realization"
            ).sum(dim="realization")

        # Number of realizations of a given driving model
        if independence_level == "model":
            realization_struct = (
                ["source", "driving_model", "experiment"]
                if balance_experiments
                else ["source", "driving_model"]
            )
        else:
            realization_struct = (
                ["driving_model", "experiment"]
                if balance_experiments
                else ["driving_model"]
            )
        realizations = {
            info[k]["member-exp"]
            for k in info.keys()
            if all([info[k][s] == sim[s] for s in realization_struct])
        }

        if skipna:
            n_realizations = len(realizations)
        else:
            n_realizations = xr.zeros_like(datasets[list(keys)[0]][v_for_skipna])
            r_models = dict()
            for r in realizations:
                r_models[r] = [
                    k
                    for k in info.keys()
                    if (
                        all([info[k][s] == sim[s] for s in realization_struct])
                        and (info[k]["member-exp"] == r)
                    )
                ]
                n_realizations = n_realizations + (
                    xr.concat(
                        [datasets[k][v_for_skipna].notnull() for k in r_models[r]],
                        dim="realization",
                    ).sum(dim="realization")
                    > 0
                )

        # Number of driving models run by a given institution
        if independence_level == "institution":
            institution_struct = (
                ["institution", "experiment"]
                if balance_experiments
                else ["institution"]
            )
            institution = {
                info[k]["driving_model"]
                for k in info.keys()
                if all([info[k][s] == sim[s] for s in institution_struct])
            }

            if skipna:
                n_institutions = len(institution)
            else:
                n_institutions = xr.zeros_like(datasets[list(keys)[0]][v_for_skipna])
                i_models = dict()
                for ii in institution:
                    i_models[ii] = [
                        k
                        for k in info.keys()
                        if (
                            all([info[k][s] == sim[s] for s in institution_struct])
                            and (info[k]["driving_model"] == ii)
                        )
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
            weights = weights.where(
                weights.experiment != e, other=weights / expsum.sel(experiment=e)
            )

        # Drop the experiment coordinate
        weights = weights.drop_vars("experiment")

    # Attribute_weights
    if attribute_weights:
        stationary_weights = {}
        non_stationary_weights = {}
        for att, v_att in attribute_weights.items():
            # Add warning when a mismatch between independance_level/experiment_weight and attribute_weights is detected
            if att == "experiment" and not balance_experiments:
                warnings.warn(
                    "Key experiment given in attribute_weights without argument balance_experiments=True"
                )

            if (
                (att == "source" and independence_level != "model")
                or (att == "driving_model" and independence_level != "GCM")
                or (att == "institution" and independence_level != "institution")
            ):
                warnings.warn(
                    f"The {att} weights do not match the {independence_level} independence_level"
                )

            # Verification
            if att not in info[k] or any(
                (info[k][att] is None or len(info[k][att]) == 0) for k in info
            ):
                raise ValueError(
                    f"The {att} attribute is missing from some simulations."
                )
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
                        raise ValueError(
                            f"The {att} {v[att]} or others are not in the attribute_weights dict."
                        )
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
                    raise ValueError(
                        f"The {att} DataArray has more than one coord dimension to apply weights."
                    )
                else:
                    coord = ls_coord[0]
                # broadcast coord to the weights DataArray
                if coord not in weights.coords:
                    weights = weights.expand_dims({coord: da[coord].values})
                ls_da = []
                for k, v in info.items():
                    if v[att] not in da[att] and "others" not in da[att]:
                        raise ValueError(
                            f"The {att} {v[att]} or others are not in the attribute_weights datarray coords."
                        )
                    elif v[att] not in da[att] and "others" in da[att]:
                        ls_da.append(da.sel(**{att: "others"}).drop_vars(att))
                    else:
                        ls_da.append(da.sel(**{att: v[att]}).drop_vars(att))
                nw = xr.concat(ls_da, weights.realization)
                weights = weights * nw

    if standardize:
        weights = weights / weights.sum(dim="realization")

    return weights


def build_partition_data(
    datasets: Union[dict, list[xr.Dataset]],
    partition_dim: list[str] = ["source", "experiment", "bias_adjust_project"],
    subset_kw: dict = None,
    regrid_kw: dict = None,
    indicators_kw: dict = None,
    rename_dict: dict = None,
):
    """Get the input for the xclim partition functions.

    From a list or dictionary of datasets, create a single dataset with
    `partition_dim` dimensions (and time) to pass to one of the xclim partition functions
    (https://xclim.readthedocs.io/en/stable/api.html#uncertainty-partitioning).
    If the inputs have different grids,
    they have to be subsetted and regridded to a common grid/point.
    Indicators can also be computed before combining the datasets.


    Parameters
    ----------
    datasets : dict
        List or dictionnary of Dataset objects that will be included in the ensemble.
        The datasets should include the necessary ("cat:") attributes to understand their metadata.
        Tip: With a project catalog, you can do: `datasets = pcat.search(**search_dict).to_dataset_dict()`.
    partition_dim: list[str]
        Components of the partition. They will become the dimension of the output.
        The default is ['source', 'experiment', 'bias_adjust_project'].
        For source, the dimension will actually be institution_source_member.
    subset_kw: dict
        Arguments to pass to `xs.spatial.subset()`.
    regrid_kw:
        Arguments to pass to `xs.regrid_dataset()`.
    indicators_kw:
        Arguments to pass to `xs.indicators.compute_indicators()`.
        All indicators have to be for the same frequency, in order to be put on a single time axis.
    rename_dict:
        Dictionary to rename the dimensions from xscen names to xclim names.
        The default is {'source': 'model', 'bias_adjust_project': 'downscaling', 'experiment': 'scenario'}.

    Returns
    -------
    xr.Dataset
        The input data for the partition functions.

    See Also
    --------
    xclim.ensembles

    """
    if isinstance(datasets, dict):
        datasets = list(datasets.values())
    # initialize dict
    subset_kw = subset_kw or {}
    regrid_kw = regrid_kw or {}

    list_ds = []
    for ds in datasets:
        if subset_kw:
            ds = subset(ds, **subset_kw)

        if regrid_kw:
            ds = regrid_dataset(ds, **regrid_kw)

        if indicators_kw:
            dict_ind = compute_indicators(ds, **indicators_kw)
            if len(dict_ind) > 1:
                raise ValueError(
                    f"The indicators computation should return only indicators of the same frequency.Returned frequencies: {dict_ind.keys()}"
                )
            else:
                ds = list(dict_ind.values())[0]

        for dim in partition_dim:
            if f"cat:{dim}" in ds.attrs:
                ds = ds.expand_dims(**{dim: [ds.attrs[f"cat:{dim}"]]})

        if "source" in partition_dim:
            new_source = f"{ds.attrs['cat:institution']}_{ds.attrs['cat:source']}_{ds.attrs['cat:member']}"
            ds = ds.assign_coords(source=[new_source])
        list_ds.append(ds)
    ens = xr.merge(list_ds)

    rename_dict = rename_dict or {}
    rename_dict.setdefault("source", "model")
    rename_dict.setdefault("experiment", "scenario")
    rename_dict.setdefault("bias_adjust_project", "downscaling")
    rename_dict = {k: v for k, v in rename_dict.items() if k in ens.dims}
    ens = ens.rename(rename_dict)

    return ens


@parse_config
def reduce_ensemble(
    data: Union[xr.DataArray, dict, list, xr.Dataset],
    method: str,
    *,
    horizons: Optional[list[str]] = None,
    create_kwargs: Optional[dict] = None,
    **kwargs,
):
    r"""Reduce an ensemble of simulations using clustering algorithms from xclim.ensembles.

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
    if isinstance(data, (list, dict)):
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

        clusters = {
            g: data.realization.isel(realization=realization[clusters_tmp == g])
            for g in np.unique(clusters_tmp)
        }
    selected = data.realization.isel(realization=selected)

    return selected, clusters, fig_data
