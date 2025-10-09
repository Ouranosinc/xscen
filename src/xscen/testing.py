"""Testing utilities for xscen."""

import importlib.metadata
import os
import re
from io import StringIO
from pathlib import Path
from typing import TextIO

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries
from xclim.testing.utils import show_versions as _show_versions


__all__ = ["datablock_3d", "fake_data", "publish_release_notes", "show_versions"]


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
    units: str | None = None,
    as_dataset: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Create a generic timeseries object based on pre-defined dictionaries of existing variables.

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
        "time": xr.DataArray(pd.date_range(start, periods=values.shape[0], freq=freq), dims="time"),
        y: xr.DataArray(
            np.arange(y_start, y_start + values.shape[1] * y_step, y_step),
            dims=y,
            attrs=attrs[y],
        )[0 : values.shape[1]],  # np.arange sometimes creates an extra value
        x: xr.DataArray(
            np.arange(x_start, x_start + values.shape[2] * x_step, x_step),
            dims=x,
            attrs=attrs[x],
        )[0 : values.shape[2]],  # np.arange sometimes creates an extra value
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
        PC = ccrs.PlateCarree()
        if x == "rlon":  # rotated pole
            GM = ccrs.RotatedPole(pole_longitude=83.0, pole_latitude=42.5, central_rotated_longitude=0.0)
            da.attrs["grid_mapping"] = "rotated_pole"
        else:
            GM = ccrs.ObliqueMercator(
                azimuth=90,
                central_latitude=46,
                central_longitude=-63,
                scale_factor=1,
                false_easting=0,
                false_northing=0,
            )
            da.attrs["grid_mapping"] = "oblique_mercator"

        YY, XX = xr.broadcast(da[y], da[x])
        pts = PC.transform_points(GM, XX.values, YY.values)
        da["lon"] = xr.DataArray(pts[..., 0], dims=XX.dims, attrs=attrs["lon"])
        da["lat"] = xr.DataArray(pts[..., 1], dims=YY.dims, attrs=attrs["lat"])

    if as_dataset:
        if "grid_mapping" in da.attrs:
            da = da.to_dataset()
            if da[variable].attrs["grid_mapping"] == "rotated_pole":
                da = da.assign_coords(
                    {
                        "rotated_pole": xr.DataArray(
                            "",
                            attrs={
                                "grid_mapping_name": "rotated_latitude_longitude",
                                "grid_north_pole_latitude": 42.5,
                                "grid_north_pole_longitude": 83.0,
                                "north_pole_grid_longitude": 0.0,
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
    """
    Generate fake data for testing.

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
    data = np.reshape(np.random.random(365 * nyears * (nx * ny)) * amplitude, (365 * nyears, ny, nx))

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
        data += np.tile(np.linspace(0, base_warming_rate * nyears, 365 * nyears) ** 3, (nx, ny, 1)).T

    # add a semi-random offset
    np.random.seed(seed)
    data = data + offset - (np.random.random() * amplitude - amplitude / 2)

    return data


def publish_release_notes(
    style: str = "md",
    file: os.PathLike | StringIO | TextIO | None = None,
    changes: str | os.PathLike | None = None,
    latest: bool = True,
) -> str | None:
    """
    Format release history in Markdown or ReStructuredText.

    Parameters
    ----------
    style : {"rst", "md"}
        Use ReStructuredText (`rst`) or Markdown (`md`) formatting. Default: Markdown.
    file : {os.PathLike, StringIO, TextIO, None}
        If provided, prints to the given file-like object. Otherwise, returns a string.
    changes : {str, os.PathLike}, optional
        If provided, manually points to the file where the changelog can be found.
        Assumes a relative path otherwise.
    latest : bool
        Whether to return the release notes of the latest version or all the content of the changelog.

    Returns
    -------
    str, optional

    Notes
    -----
    This function exists solely for development purposes. Adapted from xclim.testing.utils.publish_release_notes.
    """
    if isinstance(changes, str | Path):
        changes_file = Path(changes).absolute()
    else:
        changes_file = Path(__file__).absolute().parents[2].joinpath("CHANGELOG.rst")

    if not changes_file.exists():
        raise FileNotFoundError("Changes file not found in xscen file tree.")

    with Path(changes_file).open(encoding="utf-8") as f:
        changes = f.read()

    if style == "rst":
        hyperlink_replacements = {
            r":issue:`([0-9]+)`": r"`GH/\1 <https://github.com/Ouranosinc/xscen/issues/\1>`_",
            r":pull:`([0-9]+)`": r"`PR/\1 <https://github.com/Ouranosinc/xscen/pull/\>`_",
            r":user:`([a-zA-Z0-9_.-]+)`": r"`@\1 <https://github.com/\1>`_",
        }
    elif style == "md":
        hyperlink_replacements = {
            r":issue:`([0-9]+)`": r"[GH/\1](https://github.com/Ouranosinc/xscen/issues/\1)",
            r":pull:`([0-9]+)`": r"[PR/\1](https://github.com/Ouranosinc/xscen/pull/\1)",
            r":user:`([a-zA-Z0-9_.-]+)`": r"[@\1](https://github.com/\1)",
        }
    else:
        raise NotImplementedError()

    for search, replacement in hyperlink_replacements.items():
        changes = re.sub(search, replacement, changes)

    if latest:
        changes_split = changes.split("\n\nv0.")
        changes = changes_split[0] + "\n\nv0." + changes_split[1]

    if style == "md":
        changes = changes.replace("=========\nChangelog\n=========", "# Changelog")

        titles = {r"\n(.*?)\n([\-]{1,})": "-", r"\n(.*?)\n([\^]{1,})": "^"}
        for title_expression, level in titles.items():
            found = re.findall(title_expression, changes)
            for grouping in found:
                fixed_grouping = str(grouping[0]).replace("(", r"\(").replace(")", r"\)")
                search = rf"({fixed_grouping})\n([\{level}]{'{' + str(len(grouping[1])) + '}'})"
                replacement = f"{'##' if level == '-' else '###'} {grouping[0]}"
                changes = re.sub(search, replacement, changes)

        link_expressions = r"[\`]{1}([\w\s]+)\s<(.+)>`\_"
        found = re.findall(link_expressions, changes)
        for grouping in found:
            search = rf"`{grouping[0]} <.+>`\_"
            replacement = f"[{str(grouping[0]).strip()}]({grouping[1]})"
            changes = re.sub(search, replacement, changes)

    if not file:
        return changes
    if isinstance(file, Path | os.PathLike):
        file = Path(file).open("w")
    print(changes, file=file)


def show_versions(
    file: os.PathLike | StringIO | TextIO | None = None,
    deps: list | None = None,
) -> str | None:
    """
    Print the versions of xscen and its dependencies.

    Parameters
    ----------
    file : {os.PathLike, StringIO, TextIO}, optional
        If provided, prints to the given file-like object. Otherwise, returns a string.
    deps : list, optional
        A list of dependencies to gather and print version information from. Otherwise, prints `xscen` dependencies.

    Returns
    -------
    str or None
    """

    def _get_xscen_dependencies():
        xscen_metadata = importlib.metadata.metadata("xscen")
        requires = xscen_metadata.get_all("Requires-Dist")
        requires = [req.split("[")[0].split(";")[0].split(">")[0].split("<")[0].split("=")[0].split("!")[0] for req in requires]

        return ["xscen"] + requires

    if deps is None:
        deps = _get_xscen_dependencies()

    return _show_versions(file=file, deps=deps)
