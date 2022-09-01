import re

import numpy as np
import xarray as xr
import xclim.ensembles as xce


def build_reduction_data(
    datasets: dict, *, xrfreqs: list = None, horizons: list = None
) -> xr.DataArray:

    # Use metadata to identify the simulation attributes
    info = {}
    keys = datasets.keys() if isinstance(datasets, dict) else range(len(datasets))
    for key in keys:
        info[key] = {
            attr.replace("cat:", ""): datasets[key].attrs[attr]
            for attr in datasets[key].attrs
            if "cat:" in attr
        }

    # Find the time resolutions
    if xrfreqs is None:
        xrfreqs = np.unique(
            [
                info[k].get("xrfreq", None)
                or (xr.infer_freq(datasets[k].time) if "time" in datasets[k] else False)
                for k in info.keys()
            ]
        )

    criteria = None
    # Loop through each xrfreq
    for xrfreq in xrfreqs:
        # Subset on the datasets that have the right xrfreq and change the dictionary key to only the ID
        ds_dict = {
            info[k]["id"]: v
            for k, v in datasets.items()
            if (
                (info[k].get("xrfreq") == xrfreq)
                or (xr.infer_freq(datasets[k].time) if "time" in datasets[k] else False)
            )
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


def reduce_ensemble(data, method: str, kwargs: dict):

    selected = getattr(xce, f"{method}_reduce_ensemble")(data=data, **kwargs)

    clusters = fig_data = None
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


def _concat_criteria(criteria, ens):

    if criteria is None:
        i = 0
    else:
        i = int(criteria.criteria[-1] + 1)

    for vv in ens.data_vars:
        da = ens[vv]
        da.name = "values"
        # Stack all dimensions that are not 'realization'
        da = da.stack({"criteria": {d for d in da.dims}.difference(["realization"])})
        da = da.assign_coords({"criteria": np.arange(i, i + len(da.criteria))})
        if "horizon" in da.coords:
            da = da.drop("horizon")

        if criteria is None:
            criteria = da
        else:
            criteria = xr.concat([criteria, da], dim="criteria")
        i = i + len(da.criteria)

    return criteria
