"""Testing utilities for xscen."""

import numpy as np
import pandas as pd
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries

__all__ = ["datablock_3d"]


def datablock_3d(
    values: np.ndarray,
    variable: str,
    x: str,
    x_start: float,
    y: str,
    y_start: float,
    x_step: float = 0.1,
    y_step: float = 0.1,
    start: str = "7/1/2000",
    freq: str = "D",
    units: str = None,
    as_dataset: bool = False,
):
    """Create a generic timeseries object based on pre-defined dictionaries of existing variables.

    Parameters
    ----------
    values : np.ndarray
      The values to be assigned to the variable. Dimensions are interpreted [T, Y, X].
    variable : str
        The variable name.
    x : str
        The name of the x coordinate.
    x_start : float
        The starting value of the x coordinate.
    y : str
        The name of the y coordinate.
    y_start : float
        The starting value of the y coordinate.
    x_step : float
        The step between x values.
    y_step : float
        The step between y values.
    start : str
        The starting date of the time coordinate.
    freq : str
        The frequency of the time coordinate.
    units : str
        The units of the variable.
    as_dataset : bool
        If True, return a Dataset, else a DataArray.
    """
    attrs = {
        "lat": {
            "units": "degrees_north",
            "description": "Latitude.",
            "standard_name": "latitude",
        },
        "lon": {
            "units": "degrees_east",
            "description": "Longitude.",
            "standard_name": "longitude",
        },
        "rlon": {
            "units": "degrees",
            "description": "Rotated longitude.",
            "standard_name": "grid_longitude",
        },
        "rlat": {
            "units": "degrees",
            "description": "Rotated latitude.",
            "standard_name": "grid_latitude",
        },
        "x": {
            "units": "m",
            "description": "Projection x coordinate.",
            "standard_name": "projection_x_coordinate",
        },
        "y": {
            "units": "m",
            "description": "Projection y coordinate.",
            "standard_name": "projection_y_coordinate",
        },
    }

    dims = {
        "time": xr.DataArray(
            pd.date_range(start, periods=values.shape[0], freq=freq), dims="time"
        ),
        y: xr.DataArray(
            np.arange(y_start, y_start + values.shape[1] * y_step, y_step),
            dims=y,
            attrs=attrs[y],
        ),
        x: xr.DataArray(
            np.arange(x_start, x_start + values.shape[2] * x_step, x_step),
            dims=x,
            attrs=attrs[x],
        ),
    }

    # Get the attributes using xclim, then create the DataArray
    ts_1d = timeseries(values[:, 0, 0], variable, start, units=units, freq=freq)
    da = xr.DataArray(values, coords=dims, dims=dims, name=variable, attrs=ts_1d.attrs)

    # Add the axis information
    da[x].attrs["axis"] = "X"
    da[y].attrs["axis"] = "Y"
    da["time"].attrs["axis"] = "T"

    # Support for rotated pole and oblique mercator grids
    if x != "lon" and y != "lat":
        lat, lon = np.meshgrid(
            np.arange(45, 45 + values.shape[1] * y_step, y_step),
            np.arange(-75, -75 + values.shape[2] * x_step, x_step),
        )
        da["lat"] = xr.DataArray(np.flipud(lat.T), dims=[y, x], attrs=attrs["lat"])
        da["lon"] = xr.DataArray(lon.T, dims=[y, x], attrs=attrs["lon"])
        da.attrs["grid_mapping"] = "rotated_pole" if x == "rlon" else "oblique_mercator"

    if as_dataset:
        if "grid_mapping" in da.attrs:
            da = da.to_dataset()
            # These grid_mapping attributes are simply placeholders and won't match the data
            if da[variable].attrs["grid_mapping"] == "rotated_pole":
                da = da.assign_coords(
                    {
                        "rotated_pole": xr.DataArray(
                            "",
                            attrs={
                                "grid_mapping_name": "rotated_latitude_longitude",
                                "grid_north_pole_latitude": 42.5,
                                "grid_north_pole_longitude": 83.0,
                            },
                        )
                    }
                )
            else:
                da = da.assign_coords(
                    {
                        "oblique_mercator": xr.DataArray(
                            "",
                            attrs={
                                "grid_mapping_name": "oblique_mercator",
                                "azimuth_of_central_line": 90.0,
                                "latitude_of_projection_origin": 46.0,
                                "longitude_of_projection_origin": -63.0,
                                "scale_factor_at_projection_origin": 1.0,
                                "false_easting": 0.0,
                                "false_northing": 0.0,
                            },
                        )
                    }
                )
            return da
        else:
            return da.to_dataset()
    else:
        return da
