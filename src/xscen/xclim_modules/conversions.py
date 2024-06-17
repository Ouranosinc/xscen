"""Conversion functions for when datasets are missing particular variables and that xclim doesn't already implement."""

from __future__ import annotations  # for xclim 0.37

import xarray as xr
from xclim.core.units import convert_units_to, declare_units


@declare_units(prsn="[precipitation]", prlp="[precipitation]")
def precipitation(prsn: xr.DataArray, prlp: xr.DataArray) -> xr.DataArray:
    """Precipitation of all phases.

    Compute the precipitation flux from all phases by adding solid and liquid precipitation.

    Parameters
    ----------
    prsn: xr.DataArray
      Solid precipitation flux.
    prlp: xr.DataArray
      Liquid precipitation flux.

    Returns
    -------
    xr.DataArray, [same as prsn]
      Surface precipitation flux (all phases)
    """
    prlp = convert_units_to(prlp, prsn, context="hydro")
    pr = prsn + prlp
    pr.attrs["units"] = prsn.attrs["units"]
    return pr


@declare_units(dtr="[temperature]", tasmax="[temperature]")
def tasmin_from_dtr(dtr: xr.DataArray, tasmax: xr.DataArray) -> xr.DataArray:
    """Tasmin computed from DTR and tasmax.

    Tasmin as dtr subtracted from tasmax.

    Parameters
    ----------
    dtr: xr.DataArray
      Daily temperature range
    tasmax: xr.DataArray
      Daily maximal temperature.

    Returns
    -------
    xr.DataArray, [same as tasmax]
      Daily minimum temperature
    """
    out_units = tasmax.units
    # To prevent strange behaviors that could arise from changing DTR units, we change the units of tasmax to match DTR
    tasmax = convert_units_to(tasmax, dtr)
    tasmin = tasmax - dtr
    tasmin.attrs["units"] = dtr.units
    tasmin = convert_units_to(tasmin, out_units)
    return tasmin


@declare_units(dtr="[temperature]", tasmin="[temperature]")
def tasmax_from_dtr(dtr: xr.DataArray, tasmin: xr.DataArray) -> xr.DataArray:
    """Tasmax computed from DTR and tasmin.

    Tasmax as dtr added to tasmin.

    Parameters
    ----------
    dtr: xr.DataArray
      Daily temperature range
    tasmin: xr.DataArray
      Daily minimal temperature.

    Returns
    -------
    xr.DataArray, [same as tasmin]
      Daily maximum temperature
    """
    out_units = tasmin.attrs["units"]
    # To prevent strange behaviors that could arise from changing DTR units, we change the units of tasmin to match DTR
    tasmin = convert_units_to(tasmin, dtr)
    tasmax = tasmin + dtr
    tasmax.attrs["units"] = dtr.attrs["units"]
    tasmax = convert_units_to(tasmax, out_units)
    return tasmax


@declare_units(tasmin="[temperature]", tasmax="[temperature]")
def dtr_from_minmax(tasmin: xr.DataArray, tasmax: xr.DataArray) -> xr.DataArray:
    """DTR computed from tasmin and tasmax.

    Dtr as tasmin subtracted from tasmax.

    Parameters
    ----------
    tasmin: xr.DataArray
      Daily minimal temperature.
    tasmax: xr.DataArray
      Daily maximal temperature.

    Returns
    -------
    xr.DataArray, K
      Daily temperature range
    """
    # We force K to overcome the "delta degree" ambiguity
    tasmin = convert_units_to(tasmin, "K")
    tasmax = convert_units_to(tasmax, "K")
    dtr = tasmax - tasmin
    dtr.attrs["units"] = "K"
    return dtr
