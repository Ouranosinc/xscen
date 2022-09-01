import inspect
import logging
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
import xclim as xc
from xclim import ensembles

from .catalog import generate_id
from .config import parse_config
from .utils import clean_up

logger = logging.getLogger(__name__)

__all__ = ["ensemble_stats"]


@parse_config
def ensemble_stats(
    datasets: Union[list, dict],
    statistics: dict,
    *,
    create_kwargs: dict = None,
    ref: dict = None,
    weighted: Union[xr.DataArray, dict] = None,
    common_attrs_only: bool = True,
    to_level: str = "ensemble",
) -> xr.Dataset:
    """
    Create ensemble and calculate statistics on it.

    Parameters
    ----------
    datasets: list
        List of file paths or xarray Dataset/DataArray objects to include in the ensemble
        Tip: With a project catalog, you can do: `datasets = list(pcat.search(**search_dict).df.path)` to get a list of paths.
    create_kwargs: dict
        Dictionary of arguments for xclim.ensembles.create_ensemble.
    statistics: str
        Name of the xclim.ensemble function to call on the ensemble
        (e.g. "ensemble_percentiles","ensemble_mean_std_max_min")
    stats_kwargs: dict
        Dictionary of arguments for the statistics function.
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
        if (
            ref is not None
            and "ref" in inspect.getfullargspec(getattr(ensembles, stat))[0]
        ):
            stats_kwargs.setdefault("ref", ens.loc[ref])

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

        ens_stats = clean_up(
            ds=ens_stats, common_attrs_only=datasets, xrkwargs=create_kwargs
        )

        # generate new id
        ens_stats.attrs["cat:id"] = generate_id(ens_stats).iloc[0]

    ens_stats.attrs["cat:processing_level"] = to_level

    return ens_stats


def generate_weights(ens, info, independence_level: str = "all"):

    # TODO: Weights along the horizon dimension

    # Prepare an array of 0s, with size == nb. realization
    weights = xr.zeros_like(ens.realization, dtype=int)

    for r in ens.realization:

        # Weight == 0 means it hasn't been processed yet
        if weights.sel(realization=r) == 0:
            sim = ens.sel(realization=r)

            if ("driving_model" in info[r.item()].keys()) and not (
                str(info[r.item()].get("driving_model", None)) in (["nan", "None"])
            ):
                gcm = info[r.item()].get("driving_model", None)
            else:
                gcm = info[r.item()].get("source", None)

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
