# noqa: D100
import inspect
import logging
import warnings
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Any, Union

import numpy as np
import xarray as xr
from xclim import ensembles

from .config import parse_config
from .utils import clean_up, get_cat_attrs

logger = logging.getLogger(__name__)

__all__ = ["ensemble_stats", "generate_weights"]


@parse_config
def ensemble_stats(
    datasets: Any,
    statistics: dict,
    *,
    create_kwargs: dict = None,
    weights: xr.DataArray = None,
    common_attrs_only: bool = True,
    to_level: str = "ensemble",
) -> xr.Dataset:
    """Create an ensemble and computes statistics on it.

    Parameters
    ----------
    datasets : Any
        List of file paths or xarray Dataset/DataArray objects to include in the ensemble.
        A dictionary can be passed instead of a list, in which case the keys are used as coordinates along the new
        `realization` axis.
        Tip: With a project catalog, you can do: `datasets = pcat.search(**search_dict).to_dataset_dict()`.
    statistics : dict
        xclim.ensembles statistics to be called. Dictionary in the format {function: arguments}.
        If a function requires 'ref', the dictionary entry should be the inputs of a .loc[], e.g. {"ref": {"horizon": "1981-2010"}}
    create_kwargs : dict
        Dictionary of arguments for xclim.ensembles.create_ensemble.
    weights : xr.DataArray
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

    See Also
    --------
    xclim.ensembles._base.create_ensemble, xclim.ensembles._base.ensemble_percentiles, xclim.ensembles._base.ensemble_mean_std_max_min, xclim.ensembles._robustness.change_significance, xclim.ensembles._robustness.robustness_coefficient,

    """
    create_kwargs = create_kwargs or {}

    # if input files are .zarr, change the engine automatically
    if isinstance(datasets, list) and isinstance(datasets[0], (str, Path)):
        path = Path(datasets[0])
        if path.suffix == ".zarr" and "engine" not in create_kwargs:
            create_kwargs["engine"] = "zarr"

    ens = ensembles.create_ensemble(datasets, **create_kwargs)

    ens_stats = xr.Dataset(attrs=ens.attrs)
    for stat, stats_kwargs in statistics.items():
        stats_kwargs = deepcopy(stats_kwargs or {})
        logger.info(
            f"Creating ensemble with {len(datasets)} simulations and calculating {stat}."
        )
        if (
            weights is not None
            and "weights" in inspect.getfullargspec(getattr(ensembles, stat))[0]
        ):
            stats_kwargs["weights"] = weights.reindex_like(ens.realization)
        if "ref" in stats_kwargs:
            stats_kwargs["ref"] = ens.loc[stats_kwargs["ref"]]

        if stat == "change_significance":
            for v in ens.data_vars:
                with xr.set_options(keep_attrs=True):
                    deltak = ens[v].attrs.get("delta_kind", None)
                    if stats_kwargs.get("ref") is not None and deltak is not None:
                        raise ValueError(
                            f"{v} is a delta, but 'ref' was still specified."
                        )
                    if deltak in ["relative", "*", "/"]:
                        logging.info(
                            f"Relative delta detected for {v}. Applying 'v - 1' before change_significance."
                        )
                        ens_v = ens[v] - 1
                    else:
                        ens_v = ens[v]
                    tmp = getattr(ensembles, stat)(ens_v, **stats_kwargs)
                    if len(tmp) == 2:
                        ens_stats[f"{v}_change_frac"], ens_stats[f"{v}_pos_frac"] = tmp
                    elif len(tmp) == 3:
                        (
                            ens_stats[f"{v}_change_frac"],
                            ens_stats[f"{v}_pos_frac"],
                            ens_stats[f"{v}_p_vals"],
                        ) = tmp
                    else:
                        raise ValueError(f"Unexpected number of outputs from {stat}.")
        else:
            ens_stats = ens_stats.merge(getattr(ensembles, stat)(ens, **stats_kwargs))

    # delete the realization coordinate if there
    if "realization" in ens_stats:
        ens_stats = ens_stats.drop_vars("realization")

    # delete attrs that are not common to all dataset
    if common_attrs_only:
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


def generate_weights(
    datasets: Union[dict, list],
    *,
    independence_level: str = "model",
    experiment_weights: bool = False,
    skipna: bool = True,
    v_for_skipna: str = None,
    standardize: bool = False,
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
        'model': Weights using the method '1 model - 1 Vote', where every unique combination of 'source' and 'driving_model' is considered a model.
        'GCM': Weights using the method '1 GCM - 1 Vote'
        'institution': Weights using the method '1 institution - 1 Vote'
    experiment_weights : bool
        If True, each experiment will be given a total weight of 1. This option requires the 'cat:experiment' attribute to be present in all datasets.
    skipna : bool
        If True, weights will be computed from attributes only. If False, weights will be computed from the number of non-missing values.
        skipna=False requires either a 'time' or 'horizon' dimension in the datasets.
    v_for_skipna : str
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
                    f"Dataset {k} has dimensions that are not 'time' or 'horizon': {other_dims[k]}. The first indexes of these dimensions will be used to compute the weights."
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
    if experiment_weights and any(
        (info[k]["experiment"] is None or len(info[k]["experiment"]) == 0) for k in info
    ):
        raise ValueError(
            "The 'cat:experiment' attribute is missing from some simulations. 'experiment_weights' cannot be True."
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

        # Check that the extra dimension is the same for all datasets. If not, modify the datasets to make them the same.
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
                if experiment_weights
                else ["source", "driving_model"]
            )
        else:
            realization_struct = (
                ["driving_model", "experiment"]
                if experiment_weights
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
                ["institution", "experiment"] if experiment_weights else ["institution"]
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

    if experiment_weights:
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

    if standardize:
        weights = weights / weights.sum(dim="realization")

    return weights
