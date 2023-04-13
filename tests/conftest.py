# noqa: D100
import shutil
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

notebooks = Path().cwd().parent / "docs" / "notebooks"


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


@pytest.fixture
def tas_series_1d():
    """Create a temperature time series."""

    def _tas_series(values, start="7/1/2000", units="K", freq="D", as_dataset=True):
        coords = pd.date_range(start, periods=len(values), freq=freq)
        da = xr.DataArray(
            values,
            coords=[coords],
            dims="time",
            name="tas",
            attrs={
                "standard_name": "air_temperature",
                "cell_methods": "time: mean within days",
                "units": units,
            },
        )
        da.name = "tas"
        if as_dataset:
            return da.to_dataset()
        else:
            return da

    return _tas_series
