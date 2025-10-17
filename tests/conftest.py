# noqa: D100
import shutil
from pathlib import Path

import pytest

import xscen as xs
from xscen.testing import datablock_3d as _datablock_3d


notebooks = Path(__file__).parent.parent / "docs" / "notebooks"
SAMPLES_DIR = notebooks / "samples" / "tutorial"


@pytest.fixture(scope="session", autouse=True)
def cleanup_notebook_data_folder(request):
    """
    Cleanup a testing file once we are finished.

    This flag prevents remote data from being downloaded multiple times in the same pytest run.
    """

    def remove_data_folder():
        data = notebooks / "_data"
        if data.exists():
            shutil.rmtree(data)

    request.addfinalizer(remove_data_folder)


@pytest.fixture(scope="session")
def samplecat(request):
    """Generate a sample catalog with the tutorial netCDFs."""
    mark_skip = request.config.getoption("-m")
    if "not requires_netcdf" in mark_skip or not SAMPLES_DIR.exists():
        pytest.skip("Skipping tests that require netCDF files")
    elif not list(SAMPLES_DIR.rglob("*.nc")):
        pytest.skip("No netCDF files found in the tutorial samples folder")

    df = xs.parse_directory(
        directories=[SAMPLES_DIR],
        patterns=["{activity}/{domain}/{institution}/{source}/{experiment}/{member}/{frequency}/{?:_}.nc"],
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
    """
    Create a generic timeseries object based on pre-defined dictionaries of existing variables.

    See Also
    --------
    xscen.testing.datablock_3d
    """
    return _datablock_3d
