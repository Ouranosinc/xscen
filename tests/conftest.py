# noqa: D100
import shutil
from pathlib import Path

import pytest

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
