import inspect
import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import xarray as xr
import xclim as xc
from xclim import ensembles

from .catalog import generate_id
from .config import parse_config
from .utils import clean_up

logger = logging.getLogger(__name__)

__all__ = ["ensemble_stats", "generate_weights"]


@parse_config
def ensemble_stats(
    datasets: Any,
    statistics: dict,
    *,
    create_kwargs: dict = None,
    weighted: dict = None,
    common_attrs_only: bool = True,
    to_level: str = "ensemble",
) -> xr.Dataset:
    """
    Creates an ensemble and computes statistics on it.

    Parameters
    ----------
    datasets: Any
        List of file paths or xarray Dataset/DataArray objects to include in the ensemble.
        A dictionary can be passed instead of a list, in which case the keys are used as coordinates along the new
        `realization` axis.
        Tip: With a project catalog, you can do: `datasets = pcat.search(**search_dict).to_dataset_dict()`.
    statistics: dict
        xclim.ensembles statistics to be called. Dictionary in the format {function: arguments}.
        If a function requires 'ref', the dictionary entry should be the inputs of a .loc[], e.g. {"ref": {"horizons": "1981-2010"}}
    create_kwargs: dict
        Dictionary of arguments for xclim.ensembles.create_ensemble.
    weighted: dict
        'weights': xr.DataArray, optional
            DataArray of weights along the 'realization' dimension.
            If absent, 'info' and 'independence_level' will be used to generate weights.
        'info': dict, optional
            Dictionary in the format {realization: attrs} detailing each realization.
            If absent, the attributes will be guessed using the datasets' metadata.
        'independence_level': str
            ['all', 'GCM'] Whether to consider all simulations independent or to weight "1 GCM 1 vote".
            This entry is required unless weights are explicitely given in weighted["weights"].
    common_attrs_only:
        If True, keeps only the global attributes that are the same for all datasets and generate new id.
        If False, keeps global attrs of the first dataset (same behaviour as xclim.ensembles.create_ensemble)
    to_level: str
        The processing level to assign to the output.

    Returns
    -------
    xr.Dataset
        Dataset with ensemble statistics
    """
    create_kwargs = create_kwargs or {}
    # logger.info(f"Creating ensemble with {len(datasets)} simulations and calculating {statistics}.")

    # if input files are .zarr, change the engine automatically
    if isinstance(datasets, list) and isinstance(datasets[0], (str, Path)):
        path = Path(datasets[0])
        if path.suffix == ".zarr" and "engine" not in create_kwargs:
            create_kwargs["engine"] = "zarr"

    ens = ensembles.create_ensemble(datasets, **create_kwargs)

    if weighted is not None:

        if "weights" not in weighted.keys():
            if "info" not in weighted.keys():
                if (isinstance(datasets, list)) and (
                    isinstance(datasets[0], (str, Path))
                ):
                    raise ValueError(
                        "explicit weights are required if the dataset is a list of paths"
                    )

                # Use metadata to identify the simulation attributes
                info = {}
                keys = (
                    datasets.keys()
                    if isinstance(datasets, dict)
                    else range(len(datasets))
                )
                for key in keys:
                    info[key] = {
                        attr.replace("cat:", ""): datasets[key].attrs[attr]
                        for attr in datasets[key].attrs
                        if "cat:" in attr
                    }
            else:
                info = weighted["info"]

            weights = generate_weights(
                ens.realization, info, independence_level=weighted["independence_level"]
            )

        else:
            weights = weighted["weights"]

    ens_stats = xr.Dataset(attrs=ens.attrs)
    for stat in statistics.keys():
        stats_kwargs = statistics.get(stat, None) or {}
        if (
            weighted is not None
            and "weights" in inspect.getfullargspec(getattr(ensembles, stat))[0]
        ):
            stats_kwargs.setdefault("weights", weights)
        if "ref" in stats_kwargs:
            stats_kwargs["ref"] = ens.loc[stats_kwargs["ref"]]

        if stat == "change_significance":
            for v in ens.data_vars:
                with xc.set_options(keep_attrs=True):
                    ens_stats[f"{v}_change_frac"], ens_stats[f"{v}_pos_frac"] = getattr(
                        ensembles, stat
                    )(ens[v], **stats_kwargs)
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
            ds=ens_stats, common_attrs_only=datasets, xrkwargs=create_kwargs
        )

        # generate new id
        ens_stats.attrs["cat:id"] = generate_id(ens_stats).iloc[0]

    ens_stats.attrs["cat:processing_level"] = to_level

    return ens_stats


def generate_weights(
    ens: Union[xr.Dataset, xr.DataArray],
    info: dict = None,
    independence_level: str = "all",
) -> xr.DataArray:
    """
    Uses realization attributes to automatically generate weights along the 'realization' dimension.

    Parameters
    ----------
    ens: xr.Dataset, xr.DataArray
        Result of xclim.ensembles.create_ensemble, with datasets aligned on a dimension 'realization'.
    info: dict
        Dictionary in the format {realization: attrs} detailing each realization. Only required if independence_level != 'all'.
        The minimum required fields are 'activity' and 'source', with also 'driving_model' for regional models.
    independence_level: str
        'all': All realizations with a unique ID are considered independent.
        'GCM': Weights using the method '1 GCM - 1 Vote'

    Returns
    -------
    xr.DataArray
        Weights along the 'realization' dimension.
    """

    # TODO: 2-D weights along the horizon dimension, in the case of

    # Prepare an array of 0s, with size == nb. realization
    weights = xr.zeros_like(ens.realization, dtype=int)

    for r in ens.realization:

        # Weight == 0 means it hasn't been processed yet
        if weights.sel(realization=r) == 0:
            sim = ens.sel(realization=r)

            # Exact group corresponding to the current simulation
            group = ens.sel(
                realization=[
                    k
                    for k in info.keys()
                    if (
                        info[k].get("source", None)
                        == info[sim.realization.item()].get("source", None)
                    )
                    and (
                        info[k].get("driving_model", None)
                        == info[sim.realization.item()].get("driving_model", None)
                    )
                    and (
                        info[k].get("activity", None)
                        == info[sim.realization.item()].get("activity", None)
                    )
                ]
            )

            if independence_level == "GCM":
                if ("driving_model" in info[r.item()].keys()) and not (
                    str(info[r.item()].get("driving_model", None)) in (["nan", "None"])
                ):
                    gcm = info[r.item()].get("driving_model", None)
                else:
                    gcm = info[r.item()].get("source", None)

                # Global models
                group_g = ens.sel(
                    realization=[
                        k for k in info.keys() if info[k].get("source", None) == gcm
                    ]
                )
                # Regional models with the same GCM
                group_r = ens.sel(
                    realization=[
                        k
                        for k in info.keys()
                        if info[k].get("driving_model", None) == gcm
                    ]
                )

                # Divide the weight equally between the GCMs and RCMs
                divisor = 1 / ((len(group_g) > 0) + (len(group_r) > 0))

                # For regional models, divide between them
                if sim.realization.item() in group_r.realization:
                    divisor = divisor / len(
                        np.unique(
                            [
                                info[k.item()].get("source", None)
                                for k in group_r.realization
                            ]
                        )
                    )

            elif independence_level == "all":
                divisor = 1

            weights = weights.where(
                ~ens.realization.isin(group.realization),
                divisor / len(group.realization),
            )

    return weights
