"""Testing utilities for xscen."""
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries

__all__ = ["datablock_3d", "fake_data"]


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
    units: Optional[str] = None,
    as_dataset: bool = False,
) -> Union[xr.DataArray, xr.Dataset]:
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
    units : str, optional
        The units of the variable. If None, the units are inferred from the variable name.
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
        )[
            0 : values.shape[1]
        ],  # np.arange sometimes creates an extra value
        x: xr.DataArray(
            np.arange(x_start, x_start + values.shape[2] * x_step, x_step),
            dims=x,
            attrs=attrs[x],
        )[
            0 : values.shape[2]
        ],  # np.arange sometimes creates an extra value
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


def fake_data(
    nyears: int,
    nx: int,
    ny: int,
    rand_type: str = "random",
    seed: int = 0,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Generate fake data for testing.

    Parameters
    ----------
    nyears : int
        Number of years (365 days) to generate.
    nx : int
        Number of x points.
    ny : int
        Number of y points.
    rand_type : str
        Type of random data to generate. Options are:
        - "random": random data with no structure.
        - "tas": temperature-like data with a yearly half-sine cycle.
    seed : int
        Random seed.
    amplitude : float
        Amplitude of the random data.
    offset : float
        Offset of the random data.

    Returns
    -------
    np.ndarray
        Fake data.
    """
    if rand_type not in ["random", "tas"]:
        raise NotImplementedError(f"rand_type={rand_type} not implemented.")

    np.random.seed(seed)
    data = np.reshape(
        np.random.random(365 * nyears * (nx * ny)) * amplitude, (365 * nyears, ny, nx)
    )

    if rand_type == "tas":
        # add an annual cycle (repeating half-sine)
        data += np.tile(
            np.sin(np.linspace(0, np.pi, 365))[:, None, None] * amplitude,
            (nyears, 1, 1),
        )
        # convert to Kelvin and offset the half-sine
        data = data + 273.15 - amplitude
        # add trend (polynomial 3rd)
        np.random.seed(seed)
        base_warming_rate = 0.02 + np.random.random() * 0.01
        data += np.tile(
            np.linspace(0, base_warming_rate * nyears, 365 * nyears) ** 3, (nx, ny, 1)
        ).T

    # add a semi-random offset
    np.random.seed(seed)
    data = data + offset - (np.random.random() * amplitude - amplitude / 2)

    return data
