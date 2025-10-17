import numpy as np
from conftest import notebooks
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs
from xscen import scripting as sc


class TestScripting:
    ds = timeseries(
        np.tile(np.arange(1, 2), 50),
        variable="tas",
        start="2000-01-01",
        freq="YS-JAN",
        as_dataset=True,
    )
    ds.attrs = {
        "cat:type": "simulation",
        "cat:processing_level": "raw",
        "cat:source": "CanESM5",
        "cat:experiment": "ssp585",
        "cat:member": "r1i1p1f1",
        "cat:domain": "global",
        "cat:mip_era": "CMIP6",
        "cat:institution": "CCCma",
        "cat:activity": "ScenarioMIP",
    }

    def test_save_and_update(self):
        root = str(notebooks / "_data")

        cat = xs.ProjectCatalog.create(
            f"{root}/test_cat.json",
            project={"title": "tmp_cat", "id": "tmp"},
            overwrite=True,
        )

        first_path = root + "/test_{member}.zarr"
        sc.save_and_update(TestScripting.ds, cat, path=first_path)

        assert cat.df.path[0] == root + "/test_r1i1p1f1.zarr"
        assert cat.df.experiment[0] == "ssp585"

        sc.save_and_update(
            TestScripting.ds,
            cat,
            file_format="nc",
            # To fix hdf5 issues with h5py 3.11 on pip
            save_kwargs=dict(netcdf_kwargs={"engine": "netcdf4"}),
            build_path_kwargs={"root": root},
        )

        assert (
            cat.df.path[1]
            == root
            + "/simulation/raw/CMIP6/ScenarioMIP/global/CCCma/CanESM5/ssp585/r1i1p1f1/yr/tas/tas_yr_CMIP6_ScenarioMIP_global_CCCma_CanESM5_ssp585_r1i1p1f1_2000-2049.nc"  # noqa: E501
        )
        assert cat.df.source[1] == "CanESM5"

    def test_move_and_delete(self):
        root = str(notebooks / "_data")

        cat = xs.ProjectCatalog.create(
            f"{root}/test_cat.json",
            project={"title": "tmp_cat", "id": "tmp"},
            overwrite=True,
        )

        sc.save_and_update(TestScripting.ds, cat, path=root + "/dir1/test_r1i1p1f1.zarr")
        sc.save_and_update(TestScripting.ds, cat, path=root + "/dir1/test_r1i1p1f2.zarr")

        sc.move_and_delete(
            moving=[[root + "/dir1/test_r1i1p1f2.zarr", root + "/dir2/test_r1i1p1f2.zarr"]],
            pcat=cat,
            deleting=[root + "/dir1"],
        )
        cat.update()

        # f2 should be moved to dir2 and dir1 should be deleted (not in row 0  anymore)
        assert cat.df.path[0] == root + "/dir2/test_r1i1p1f2.zarr"
        assert len(cat.df) == 1  # only one file left

        sc.move_and_delete(
            moving=[[root + "/dir2/test_r1i1p1f2.zarr", root + "/dir1/test_r1i1p1f2.zarr"]],
            pcat=cat,
            copy=True,
        )
        cat.update()

        # f2 should be copied to dir1 and f1 should still exist
        assert cat.df.path[0] == root + "/dir2/test_r1i1p1f2.zarr"
        assert cat.df.path[1] == root + "/dir1/test_r1i1p1f2.zarr"
        assert len(cat.df) == 2  # only one file left
