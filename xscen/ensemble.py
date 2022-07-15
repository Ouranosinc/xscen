import logging

from xclim import ensembles

from .catalog import ProjectCatalog

logger = logging.getLogger(__name__)


def ensemble_stats(
    files: list,
    create_args: dict = {},
    statistics: str = "ensemble_percentiles",
    stats_args: dict = {},
):
    """
    Create ensemble and calculate statistics on it.

    Parameters
    ----------
    files: list
        List of files to include in the ensemble
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
        Dataset with ensemble statitics
    """
    logger.info(f"Creating ensemble with {len(files)} and calculating {statistics}.")
    ens = ensembles.create_ensemble(files, **create_args)
    ens_stats = getattr(ensembles, statistics)(ens, **stats_args)
    return ens_stats
