# noqa: D100
import shutil
from functools import partial
from pathlib import Path

import pytest
from xclim.testing.helpers import test_timeseries

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
def tas_series():
    """Return mean temperature time series."""
    _tas_series = partial(test_timeseries, variable="tas")
    return _tas_series
