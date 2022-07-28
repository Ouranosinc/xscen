import logging
from pathlib import Path
# from typing import Optional, Union

import pandas as pd
import xarray as xr
from xclim import ensembles

from .catalog import generate_id  # ProjectCatalog

logger = logging.getLogger(__name__)

__all__ = ["ensemble_stats"]


def ensemble_stats(
    datasets: list,
    create_kwargs: dict = None,
    statistics: str = "ensemble_percentiles",
    stats_kwargs: dict = None,
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
    if stats_kwargs is None:
        stats_kwargs = {}
    if create_kwargs is None:
        create_kwargs = {}
    logger.info(f"Creating ensemble with {len(datasets)} and calculating {statistics}.")

    # if input files are .zarr, change the engine automatically
    if isinstance(datasets[0], (str, Path)):
        if len(datasets) > 1:
            create_kwargs.setdefault("mf_flag", True)
        path = Path(datasets[0])
        if path.suffix == ".zarr" and "engine" not in create_kwargs:
            create_kwargs["engine"] = "zarr"

    ens = ensembles.create_ensemble(datasets, **create_kwargs)
    ens_stats = getattr(ensembles, statistics)(ens, **stats_kwargs)

    # delete attrs that are not common to all dataset
    if common_attrs_only:
        for i in range(len(datasets)):
            if isinstance(datasets[i], (str, Path)):
                # if they exist remove attrs specific to create_ensemble
                create_kwargs.pop("mf_flag", None)
                create_kwargs.pop("resample_freq", None)
                create_kwargs.pop("calendar", None)
                ds = xr.open_dataset(datasets[i], **create_kwargs)
            else:
                ds = datasets[i]
            attributes = ens_stats.attrs.copy()
            for a_key, a_val in attributes.items():
                if (
                    (a_key not in ds.attrs)
                    or (a_key in ["cat/date_start", "cat/date_end"])
                    or (a_val != ds.attrs[a_key])
                ):
                    del ens_stats.attrs[a_key]
        # create dataframe of catalogue attrs to generate new id
        df = pd.DataFrame.from_dict(
            {
                key[4:]: [value]
                for key, value in ens_stats.attrs.items()
                if key[:4] == "cat/"
            }
        )
        ens_stats.attrs["cat/id"] = generate_id(df)[0]

    ens_stats.attrs["cat/processing_level"] = to_level

    return ens_stats
