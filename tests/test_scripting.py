import shutil as sh

import numpy as np
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs

from .conftest import notebooks


class TestScripting:
    ds = timeseries(
        np.tile(np.arange(1, 2), 50),
        variable="tas",
        start="2000-01-01",
        freq="AS-JAN",
        as_dataset=True,
    )
    ds.attrs = {
        "cat:type": "simulation",
        "cat:source": "CanESM5",
        "cat:experiment": "ssp585",
        "cat:member": "r1i1p1f1",
    }

    def test_save_and_update(samplecat):
        root = notebooks / "_data"
        first_path = root + "/test_{member}.zarr"
        xs.save_and_update(TestScripting.ds, samplecat, path=first_path)

        assert samplecat.df.path[0] == root + "/test_r1i1p1f1.zarr"
        assert samplecat.df.experiment[0] == "ssp585"

        xs.save_and_update(
            TestScripting.ds,
            samplecat,
            file_format="nc",
            build_path_kwargs={"root": root},
        )

        assert (
            samplecat.df.path[1]
            == root
            + "/simulation/CanESM5/ssp585/r1i1p1f1/tas_Amon_CanESM5_ssp585_r1i1p1f1_gn_200001-204912.nc"
        )
        assert samplecat.df.source[1] == "CanESM5"

    # def test_move_and_delete(self):
