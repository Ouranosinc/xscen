# noqa: D100
import inspect
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Union

import pandas as pd
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
    stats_kwargs=None,
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
        If a function requires 'ref', the dictionary entry should be the inputs of a .loc[], e.g. {"ref": {"horizons": "1981-2010"}}
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

    if isinstance(statistics, str) and isinstance(stats_kwargs, dict):
        warnings.warn(
            "The usage of 'statistics: str' with 'stats_kwargs: dict' will be abandoned. "
            "Please use 'statistics: dict' instead.",
            category=FutureWarning,
        )
        statistics = {statistics: stats_kwargs}
        stats_kwargs = None

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
                            "{v} is a delta, but 'ref' was still specified."
                        )
                    if deltak in ["relative", "*", "/"]:
                        logging.info(
                            "Relative delta detected for {v}. Applying 'v - 1' before change_significance."
                        )
                        ens_v = ens[v] - 1
                    else:
                        ens_v = ens[v]
                    ens_stats[f"{v}_change_frac"], ens_stats[f"{v}_pos_frac"] = getattr(
                        ensembles, stat
                    )(ens_v, **stats_kwargs)
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
    independence_level: str = "all",
) -> xr.DataArray:
    """Use realization attributes to automatically generate weights along the 'realization' dimension.

    Parameters
    ----------
    datasets : dict
        List of Dataset objects that will be included in the ensemble.
        The datasets should include attributes to help recognize them - 'cat:activity','cat:source', and 'cat:driving_model' for regional models.
        A dictionary can be passed instead of a list, in which case the keys are used for the 'realization' coordinate.
        Tip: With a project catalog, you can do: `datasets = pcat.search(**search_dict).to_dataset_dict()`.
    independence_level : str
        'all': Weights using the method '1 model - 1 Vote', where every unique combination of 'source' and 'driving_model' is considered a model.
        'GCM': Weights using the method '1 GCM - 1 Vote'

    Returns
    -------
    xr.DataArray
        Weights along the 'realization' dimension.
    """
    # TODO: 2-D weights along the horizon dimension

    if (isinstance(datasets, list)) and (isinstance(datasets[0], (str, Path))):
        raise ValueError(
            "explicit weights are required if the dataset is a list of paths"
        )

    # Use metadata to identify the simulation attributes
    keys = datasets.keys() if isinstance(datasets, dict) else range(len(datasets))
    defdict = {"source": None, "activity": None, "driving_model": None}
    info = {key: dict(defdict, **get_cat_attrs(datasets[key])) for key in keys}

    # Prepare an array of 0s, with size == nb. realization
    weights = xr.DataArray(
        [0] * len(keys), coords={"realization": ("realization", list(keys))}
    )

    for r in weights.realization.values:
        # Weight == 0 means it hasn't been processed yet
        if weights.sel(realization=r) == 0:
            sim = info[r]

            # Exact group corresponding to the current simulation
            group = [
                k
                for k in info.keys()
                if (info[k].get("source", None) == sim.get("source", None))
                and (
                    info[k].get("driving_model", None) == sim.get("driving_model", None)
                )
                and (info[k].get("activity", None) == sim.get("activity", None))
            ]

            if independence_level == "GCM":
                if ("driving_model" in sim.keys()) and not (
                    pd.isna(sim.get("driving_model"))
                ):
                    gcm = sim.get("driving_model", None)
                else:
                    gcm = sim.get("source", None)

                # Global models
                group_g = [k for k, v in info.items() if v.get("source") == gcm]
                # Regional models with the same GCM
                group_r = [k for k, v in info.items() if v.get("driving_model") == gcm]

                # Divide the weight equally between the GCMs and RCMs
                divisor = 1 / ((len(group_g) > 0) + (len(group_r) > 0))

                # For regional models, divide between them
                if r in group_r:
                    divisor = divisor / len({info[k].get("source") for k in group_r})

            elif independence_level == "all":
                divisor = 1

            else:
                raise ValueError(
                    f"'independence_level' should be between 'GCM' and 'all', received {independence_level}."
                )

            weights = weights.where(
                ~weights.realization.isin(group),
                divisor / len(group),
            )

    return weights
