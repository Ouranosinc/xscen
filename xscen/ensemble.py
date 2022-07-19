import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import xarray as xr
from xclim import ensembles

from .catalog import ProjectCatalog, generate_id

logger = logging.getLogger(__name__)


def ensemble_stats(
    datasets: list,
    create_args: dict = {},
    statistics: str = "ensemble_percentiles",
    stats_args: dict = {},
    attrs_to_keep: Optional[Union[list, str]] = None,
    to_level: str = "ensemble",
):
    """
    Create ensemble and calculate statistics on it.

    Parameters
    ----------
    datasets: list
        List of file paths or xarray Dataset/DataArray objects to include in the ensemble
        Tip: With a project catalog, you can do: `datasets = list(pcat.search(**search_dict).df.path)`.
    create_args: dict
        Dictionary of arguments for xclim.ensembles.create_ensemble
    statistics: str
        Name of the xclim.ensemble function to call on the ensemble
        (eg. "ensemble_percentiles","ensemble_mean_std_max_min").
    stats_args: dict
        Dictionary of arguments for the statictics function
    attrs_to_keep: list, str
        List of global attributes that fit all members of the ensemble and that will be kept in the output
        If none is given the default is:
        ["cat/type","cat/bias_adjust_institution", "cat/bias_adjust_project","cat/xrfreq", "cat/frequency",
        "cat/experiment","cat/domain","cat/date_start", "cat/date_end"].
        If 'all', all attributes will be kept.
    to_level: str
        The processing level to assign to the output.

    Returns
    -------
    ens_stats: xr.Dataset
        Dataset with ensemble statistics
    """
    logger.info(f"Creating ensemble with {len(datasets)} and calculating {statistics}.")

    # if input files are .zarr, change the engine automatically
    if isinstance(datasets[0], (str, Path)):
        path = Path(datasets[0])
        if path.suffix == ".zarr" and "engine" not in create_args:
            create_args["engine"] = "zarr"

    ens = ensembles.create_ensemble(datasets, **create_args)
    ens_stats = getattr(ensembles, statistics)(ens, **stats_args)

    # delete attrs that were copied from the first dataset and only put back the ones that apply to the whole ensemble
    if attrs_to_keep != "all":
        ens_stats.attrs = {}
        if isinstance(datasets[0], (str, Path)):
            ds = xr.open_dataset(datasets[0], **create_args)
        else:
            ds = datasets[0]

        ATTRIBUTES = [
            "cat/type",
            "cat/bias_adjust_institution",
            "cat/bias_adjust_project",
            "cat/xrfreq",
            "cat/frequency",
            "cat/experiment",
            "cat/domain",
            "cat/date_start",
            "cat/date_end",
        ]

        attrs_to_keep = attrs_to_keep or ATTRIBUTES

        df = pd.DataFrame()
        for a in attrs_to_keep:
            a_nocat = a.replace("cat/", "")
            ens_stats.attrs[a] = ds.attrs[a] if a in ds.attrs else None
            df[a_nocat] = [ds.attrs[a] if a in ds.attrs else None]

        ens_stats.attrs["cat/processing_level"] = to_level
        ens_stats.attrs["cat/id"] = generate_id(df)[0]

    else:
        ens_stats.attrs["cat/processing_level"] = to_level

    return ens_stats
