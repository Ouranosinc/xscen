import logging
from pathlib import Path

from xclim import ensembles

from .catalog import ProjectCatalog

logger = logging.getLogger(__name__)


def ensemble_stats(
    datasets: list,
    create_args: dict = {},
    statistics: str = "ensemble_percentiles",
    stats_args: dict = {},
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

    Returns
    -------
    ens_stats: xr.Dataset
        Dataset with ensemble statistics
    """
    logger.info(f"Creating ensemble with {len(datasets)} and calculating {statistics}.")

    # if input files are .zarr, change the engine automatically
    if isinstance(datasets[0], (str, Path)):
        path = Path(datasets[0])
        if path.suffix == ".zarr":
            if "xr_kwargs" not in create_args:
                create_args["xr_kwargs"] = {}
            create_args["xr_kwargs"]["engine"] = "zarr"

    ens = ensembles.create_ensemble(datasets, **create_args)
    ens_stats = getattr(ensembles, statistics)(ens, **stats_args)
    return ens_stats
