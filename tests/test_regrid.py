import numpy as np
import pytest

try:
    import xesmf as xe
except ImportError:
    xe = None
from xscen.regrid import create_bounds_rotated_pole, regrid_dataset
from xscen.testing import datablock_3d


def test_create_bounds_rotated_pole():
    ds = datablock_3d(
        np.zeros((20, 10, 10)),
        "tas",
        "rlon",
        -5,
        "rlat",
        80.5,
        1,
        1,
        "2000-01-01",
        as_dataset=True,
    )
    bnds = create_bounds_rotated_pole(ds)
    np.testing.assert_allclose(bnds.lon_bounds[-1, -1, 1], 83)
    np.testing.assert_allclose(bnds.lat_bounds[-1, -1, 1], 42.5)


@pytest.mark.skipif(xe is None, reason="xesmf needed for testing regrdding")
class TestRegridDataset:
    def test_simple(self, tmp_path):
        dsout = datablock_3d(
            np.zeros((2, 10, 10)),
            "tas",
            "rlon",
            -5,
            "rlat",
            -5,
            1,
            1,
            "2000-01-01",
            as_dataset=True,
        )
        dsout.attrs["cat:domain"] = "Région d'essai"

        dsin = datablock_3d(
            np.zeros((10, 6, 6)),
            "tas",
            "lon",
            -142,
            "lat",
            0,
            2,
            2,
            "2000-01-01",
            as_dataset=True,
        )
        dsin = dsin.chunk({"lon": 3, "time": 1})

        out = regrid_dataset(
            dsin,
            dsout,
            tmp_path / "weights",
            regridder_kwargs={
                "method": "patch",
                "output_chunks": {"rlon": 5},
                "unmapped_to_nan": True,
            },
        )

        assert (tmp_path / "weights" / "weights_regrid0patch.nc").is_file()
        assert out.tas.attrs["grid_mapping"] == "rotated_pole"
        assert out.rotated_pole.attrs == dsout.rotated_pole.attrs
        assert "patch" in out.attrs["history"]
        assert out.attrs["cat:processing_level"] == "regridded"
        assert out.chunks["rlon"] == (5, 5)
