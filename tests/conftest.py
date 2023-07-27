# noqa: D100
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs

notebooks = Path(__file__).parent.parent / "docs" / "notebooks"
SAMPLES_DIR = notebooks / "samples" / "tutorial"


@pytest.fixture(scope="session", autouse=True)
def cleanup_notebook_data_folder(request):
    """Cleanup a testing file once we are finished.

    This flag prevents remote data from being downloaded multiple times in the same pytest run.
    """

    def remove_data_folder():
        data = notebooks / "_data"
        if data.exists():
            shutil.rmtree(data)

    request.addfinalizer(remove_data_folder)


@pytest.mark.requires_docs
@pytest.fixture(scope="session")
def samplecat():
    """Generate a sample catalog with the tutorial netCDFs."""
    df = xs.parse_directory(
        directories=[SAMPLES_DIR],
        patterns=[
            "{activity}/{domain}/{institution}/{source}/{experiment}/{member}/{frequency}/{?:_}.nc"
        ],
        homogenous_info={
            "mip_era": "CMIP6",
            "type": "simulation",
            "processing_level": "raw",
        },
        read_from_file=["variable", "date_start", "date_end"],
        xr_open_kwargs={"engine": "h5netcdf"},
    )
    return xs.DataCatalog({"esmcat": xs.catalog.esm_col_data, "df": df})


@pytest.fixture
def datablock_3d():
    def _datablock_3d(
        values,
        variable,
        x,
        x_start,
        y,
        y_start,
        x_step=0.1,
        y_step=0.1,
        start="7/1/2000",
        freq="D",
        units=None,
        as_dataset=False,
    ):
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
        da = xr.DataArray(
            values, coords=dims, dims=dims, name=variable, attrs=ts_1d.attrs
        )

        # Add the axis information
        da[x].attrs["axis"] = "X"
        da[y].attrs["axis"] = "Y"
        da["time"].attrs["axis"] = "T"

        # TODO: Fully support rotated grids (2D coordinates + grid_mapping)

        if as_dataset:
            return da.to_dataset()
        else:
            return da

    return _datablock_3d
