"""Functions to reduce an ensemble of simulations."""

import warnings

import numpy as np
import xarray as xr
import xclim.ensembles as xce

from .config import parse_config


@parse_config
def build_reduction_data(
    datasets: dict | list[xr.Dataset],
    *,
    xrfreqs: list[str] | None = None,
    horizons: list[str] | None = None,
) -> xr.DataArray:
    """Construct the input required for ensemble reduction.

    This will combine all variables into a single DataArray and stack all dimensions except "realization".

    Parameters
    ----------
    datasets : Union[dict, list]
        Dictionary of datasets in the format {"id": dataset}, or list of datasets. This can be generated by calling .to_dataset_dict() on a catalog.
    xrfreqs : list of str, optional
        List of unique frequencies across the datasets.
        If None, the script will attempt to guess the frequencies from the datasets' metadata or with xr.infer_freq().
    horizons : list of str, optional
        Subset of horizons on which to create the data.

    Returns
    -------
    xr.DataArray
        2D DataArray of dimensions "realization" and "criteria", to be used as input for ensemble reduction.
    """
    warnings.warn(
        "This function will be dropped in v0.11.0, as it is now redundant with xclim.ensembles.make_criteria."
        "Either use xclim.ensembles.make_criteria directly (preceded by xclim.ensembles.create_ensemble if needed) or "
        "use xscen's reduce_ensemble function to build the criteria and reduce the ensemble in one step.",
        FutureWarning,
    )

    # Use metadata to identify the simulation attributes
    info = {}
    keys = datasets.keys() if isinstance(datasets, dict) else range(len(datasets))
    for key in keys:
        info[key] = {}
        info[key]["id"] = datasets[key].attrs.get("cat:id", None) or key
        info[key]["xrfreq"] = datasets[key].attrs.get("cat:xrfreq") or xr.infer_freq(
            datasets[key].time
        )

    xrfreqs = xrfreqs or np.unique(info[key]["xrfreq"] for key in info.keys())

    criteria = None
    # Loop through each xrfreq
    for xrfreq in xrfreqs:
        # Subset on the datasets that have the right xrfreq and change the dictionary key to only the ID
        ds_dict = {
            info[k]["id"]: v for k, v in datasets.items() if info[k]["xrfreq"] == xrfreq
        }

        # Create the ensemble
        ens = xce.create_ensemble(datasets=ds_dict)

        if horizons:
            ens = ens.where(ens.horizon.isin(horizons), drop=True)

        criteria = _concat_criteria(criteria, ens)

    # drop columns that are all NaN
    criteria = criteria.dropna(dim="criteria", how="all")
    if criteria.isnull().sum().values != 0:
        raise ValueError("criteria dataset contains NaNs")

    # Attributes
    criteria.attrs = {"long_name": "criteria for ensemble selection"}

    return criteria


@parse_config
def reduce_ensemble(
    data: xr.DataArray | dict | list | xr.Dataset,
    method: str,
    *,
    horizons: list[str] | None = None,
    create_kwargs: dict | None = None,
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
    warnings.warn(
        "This function has been moved to xscen.ensembles.reduce_ensemble. This version will be dropped in v0.11.0.",
        FutureWarning,
    )
    return reduce_ensemble(
        data=data,
        method=method,
        horizons=horizons,
        create_kwargs=create_kwargs,
        **kwargs,
    )


def _concat_criteria(criteria: xr.DataArray | None, ens: xr.Dataset):
    """Combine all variables and dimensions excepting 'realization'."""
    if criteria is None:
        i = 0
    else:
        i = int(criteria.criteria[-1] + 1)

    for vv in ens.data_vars:
        da = ens[vv]
        da.name = "values"
        # Stack all dimensions that are not 'realization'
        da = da.stack(
            {"criteria": list({d for d in da.dims}.difference(["realization"]))}
        )
        da = da.assign_coords({"criteria": np.arange(i, i + len(da.criteria))})
        if "horizon" in da.coords:
            da = da.drop_vars("horizon")

        if criteria is None:
            criteria = da
        else:
            criteria = xr.concat([criteria, da], dim="criteria")
        i = i + len(da.criteria)

    return criteria
