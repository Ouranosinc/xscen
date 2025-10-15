import hashlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


try:
    import xesmf as xe
except (ImportError, KeyError) as e:
    if isinstance(e, KeyError):
        if e.args[0] != "Author":
            raise e
    xe = None
import xscen as xs
from xscen.testing import datablock_3d


class TestCreateBoundsGridmapping:
    def test_create_bounds_rotated_pole(self):
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
        bnds = xs.regrid.create_bounds_gridmapping(ds, "rotated_pole")
        np.testing.assert_allclose(bnds.lon_bounds[-1, -1, 1], 83)
        np.testing.assert_allclose(bnds.lat_bounds[-1, -1, 1], 42.5)

        with pytest.warns(FutureWarning):
            assert xs.regrid.create_bounds_rotated_pole(ds).equals(bnds)

    def test_create_bounds_crs(self):
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
        ds = ds.rename({"rotated_pole": "crs"})
        ds.tas.attrs["grid_mapping"] = "crs"
        bnds = xs.regrid.create_bounds_gridmapping(ds)
        np.testing.assert_allclose(bnds.lon_bounds[-1, -1, 1], 83)
        np.testing.assert_allclose(bnds.lat_bounds[-1, -1, 1], 42.5)

    def test_create_bounds_oblique(self):
        ds = datablock_3d(
            np.zeros((20, 10, 10)),
            "tas",
            "x",
            -5000,
            "y",
            5000,
            100000,
            100000,
            "2000-01-01",
            as_dataset=True,
        )
        bnds = xs.regrid.create_bounds_gridmapping(ds, "oblique_mercator")
        np.testing.assert_allclose(bnds.lon_bounds[-1, -1, -1], -48.98790806)
        np.testing.assert_allclose(bnds.lat_bounds[-1, -1, -1], 52.9169163)

    def test_error(self):
        ds = datablock_3d(
            np.zeros((20, 10, 10)),
            "tas",
            "x",
            -5000,
            "y",
            5000,
            100000,
            100000,
            "2000-01-01",
            as_dataset=True,
        )
        ds = ds.rename({"oblique_mercator": "lambert_conformal_conic"})
        ds["lambert_conformal_conic"].attrs["grid_mapping_name"] = "lambert_conformal_conic"
        ds.tas.attrs["grid_mapping"] = "lambert_conformal_conic"
        with pytest.raises(NotImplementedError):
            xs.regrid.create_bounds_gridmapping(ds, "lambert_conformal_conic")

    def test_error_gridmap(self):
        ds = datablock_3d(
            np.zeros((20, 10, 10)),
            "tas",
            "x",
            -5000,
            "y",
            5000,
            100000,
            100000,
            "2000-01-01",
            as_dataset=True,
        )
        with pytest.raises(ValueError):
            xs.regrid.create_bounds_gridmapping(ds, "lambert_conformal_conic")


@pytest.mark.skipif(xe is None, reason="xesmf needed for testing regrdding")
class TestRegridDataset:
    @staticmethod
    def compute_file_hash(file_path):
        """Compute the SHA-256 hash of the specified file."""
        sha256 = hashlib.sha256()
        with Path(file_path).open("rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()

    dsin_reg = datablock_3d(
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
    dsin_reg = dsin_reg.chunk({"lon": 3, "time": 1})
    dsin_reg.attrs["cat:id"] = "CMIP5"
    dsin_reg.attrs["cat:member"] = "MIP5"
    dsin_reg.attrs["cat:domain"] = "Global"

    dsout_rp = datablock_3d(
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
    dsout_rp.attrs["cat:id"] = "CORDEX"
    dsout_rp.attrs["cat:domain"] = "RegionEssai"

    def test_simple(self, tmp_path):
        out = xs.regrid_dataset(
            self.dsin_reg,
            self.dsout_rp,
            weights_location=tmp_path / "weights",
            regridder_kwargs={
                "method": "patch",
                "output_chunks": {"rlon": 5},
                "unmapped_to_nan": True,
            },
        )

        assert (tmp_path / "weights" / "C_Global_CORDEX_RegionEssai_regrid0patch.nc").is_file()
        assert out.tas.attrs["grid_mapping"] == "rotated_pole"
        assert out.rotated_pole.attrs == self.dsout_rp.rotated_pole.attrs
        assert "patch" in out.attrs["history"]
        assert out.attrs["cat:processing_level"] == "regridded"
        assert out.chunks["rlon"] == (5, 5)

        hash1 = self.compute_file_hash(tmp_path / "weights" / "C_Global_CORDEX_RegionEssai_regrid0patch.nc")
        xs.regrid_dataset(
            self.dsin_reg,
            self.dsout_rp,
            weights_location=tmp_path / "weights",
            regridder_kwargs={
                "method": "patch",
                "output_chunks": {"rlon": 5},
                "unmapped_to_nan": True,
            },
        )
        # Check that the weights are not recomputed
        hash2 = self.compute_file_hash(tmp_path / "weights" / "C_Global_CORDEX_RegionEssai_regrid0patch.nc")
        assert hash1 == hash2

    def test_mask(self):
        ds_in = self.dsin_reg.copy()
        ds_in["tas"].loc[dict(lon=-142, lat=0)] = 999999
        ds_in["mask"] = xr.ones_like(ds_in.tas.isel(time=0))
        ds_in["mask"].loc[dict(lon=-142, lat=0)] = 0

        grid = xe.util.cf_grid_2d(-140, -134, 1, 2, 8, 1)
        grid["mask"] = xr.DataArray(np.ones((6, 6)), dims=("lat", "lon"))
        grid["mask"].loc[dict(lon=-134.5, lat=7.5)] = 0

        out = xs.regrid_dataset(
            ds_in,
            grid,
            regridder_kwargs={
                "method": "bilinear",
                "skipna": True,
            },
        )
        assert "mask" not in out
        np.testing.assert_equal(out.tas.isel(time=0), xr.where(grid["mask"] == 0, np.nan, 0).values)

    @pytest.mark.parametrize(
        "unmapped_to_nan, skipna",
        [[True, False], [False, True], [False, False], [None, False]],
    )
    def test_unmapped_to_nan(self, unmapped_to_nan, skipna):
        out = xs.regrid_dataset(
            self.dsin_reg,
            xe.util.cf_grid_2d(-140, -130, 1, 2, 8, 1),
            regridder_kwargs={
                "method": "bilinear",
                "skipna": skipna,
                "unmapped_to_nan": unmapped_to_nan,
            },
        )
        if skipna is False and not unmapped_to_nan:
            # This is the only case where unmapped NaNs will be extrapolated
            np.testing.assert_equal(out.tas.isel(time=0, lat=0), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        else:
            np.testing.assert_equal(
                out.tas.isel(time=0, lat=0),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, np.nan, np.nan]),
            )

    def test_no_regrid(self, tmp_path):
        out = xs.regrid_dataset(
            self.dsin_reg,
            self.dsin_reg,
            regridder_kwargs={
                "method": "patch",
                "output_chunks": {"rlon": 5},
                "unmapped_to_nan": True,
            },
            to_level="regridded2",
        )
        assert out.equals(self.dsin_reg)
        assert out.attrs["cat:processing_level"] == "regridded2"

        # Add a mask
        ds_in = self.dsin_reg.copy()
        ds_in["mask"] = xr.ones_like(ds_in.tas.isel(time=0))
        ds_in["mask"].loc[dict(lon=-142, lat=0)] = 0
        ds_grid = self.dsin_reg.copy()
        ds_grid["mask"] = xr.ones_like(ds_grid.tas.isel(time=0))
        ds_grid["mask"].loc[dict(lon=-142, lat=2)] = 0
        out = xs.regrid_dataset(
            ds_in,
            ds_grid,
            regridder_kwargs={
                "method": "patch",
                "output_chunks": {"rlon": 5},
                "unmapped_to_nan": True,
            },
        )
        np.testing.assert_allclose(out.tas.isel(time=2, lon=0), np.array([np.nan, np.nan, 0, 0, 0, 0]))
        np.testing.assert_allclose(out.tas.isel(time=2, lon=1), np.array([0, 0, 0, 0, 0, 0]))
        assert "mask" not in out

    def test_intermediate(self, tmp_path):
        intermediate = {
            "intermediate": {
                "cf_grid_2d": {
                    "lon0_b": -142,
                    "lon1_b": -132,
                    "d_lon": 1,
                    "lat0_b": 0,
                    "lat1_b": 10,
                    "d_lat": 1,
                },
                "regridder_kwargs": {"method": "bilinear", "skipna": True},
            }
        }
        out = xs.regrid_dataset(
            self.dsin_reg,
            self.dsout_rp,
            weights_location=tmp_path / "weights",
            intermediate_grids=intermediate,
            regridder_kwargs={
                "method": "patch",
                "output_chunks": {"rlon": 5},
                "unmapped_to_nan": True,
            },
        )
        assert (tmp_path / "weights" / "C_Global_CORDEX_RegionEssai_regrid0bilinear.nc").is_file()
        assert (tmp_path / "weights" / "C_Global_CORDEX_RegionEssai_regrid1patch.nc").is_file()
        assert "cf_grid_2d with arguments" in out.attrs["history"]

    @pytest.mark.parametrize("gridmap", ["oblique_mercator", "rotated_pole"])
    def test_conservative_in(self, tmp_path, gridmap):
        mult = 1 if gridmap == "rotated_pole" else 100000

        ds_in = datablock_3d(
            np.tile(np.arange(6), (1, 4, 1)) / 86400,
            "pr",
            "rlon" if gridmap == "rotated_pole" else "x",
            0,
            "rlat" if gridmap == "rotated_pole" else "y",
            0,
            1 * mult,
            1 * mult,
            "2000-01-01",
            as_dataset=True,
            units="kg m-2 s-1",
        )
        grid = xe.util.cf_grid_2d(
            ds_in.lon.min().values - 0.5,
            ds_in.lon.max().values + 0.5,
            0.22,
            ds_in.lat.min().values - 0.5,
            ds_in.lat.max().values + 0.5,
            0.22,
        )

        out = xs.regrid_dataset(
            ds_in,
            grid,
            regridder_kwargs={
                "method": "conservative",
                "skipna": False,
            },
        )
        assert out.attrs["regrid_method"] == "conservative"
        assert "bounds" not in out.dims
        assert "lon_bounds" not in out
        assert "lat_bounds" not in out
        assert gridmap not in out

    def test_conservative_out(self, tmp_path):
        ds_in = datablock_3d(
            np.tile(np.arange(6), (1, 4, 1)) / 86400,
            "pr",
            "lon",
            -142.5,
            "lat",
            2,
            5,
            1,
            "2000-01-01",
            as_dataset=True,
            units="kg m-2 s-1",
        )

        out = xs.regrid_dataset(
            ds_in,
            self.dsout_rp,
            regridder_kwargs={
                "method": "conservative",
                "skipna": False,
            },
        )
        assert out.attrs["regrid_method"] == "conservative"
        assert "bounds" not in out.dims
        assert "lon_bounds" not in out
        assert "lat_bounds" not in out
        assert "rotated_pole" in out.coords
        assert all(d in out.dims for d in ["rlon", "rlat"])
        assert all(c in out.coords for c in ["lon", "lat"])
        assert out.pr.attrs["grid_mapping"] == "rotated_pole"

    def test_conservative_multiple(self):
        ds_in = datablock_3d(
            np.tile(np.arange(6), (1, 4, 1)) / 86400,
            "pr",
            "lon",
            -142.5,
            "lat",
            2,
            5,
            1,
            "2000-01-01",
            as_dataset=True,
            units="kg m-2 s-1",
        )

        out = xs.regrid_dataset(
            ds_in,
            self.dsout_rp,
            regridder_kwargs={
                "method": "conservative",
                "skipna": False,
            },
        )
        assert out.attrs["regrid_method"] == "conservative"
        assert "bounds" not in out.dims
        assert "lon_bounds" not in out
        assert "lat_bounds" not in out
        assert "rotated_pole" in out.coords
        assert all(d in out.dims for d in ["rlon", "rlat"])
        assert all(c in out.coords for c in ["lon", "lat"])
        assert out.pr.attrs["grid_mapping"] == "rotated_pole"


class TestMask:
    ds = datablock_3d(
        np.tile(np.arange(6), (1, 4, 1)),
        "tas",
        "lon",
        -142.5,
        "lat",
        2,
        5,
        1,
        "2000-01-01",
        as_dataset=True,
    )
    ds["tas"] = ds["tas"].where(ds["tas"].lon > -142.5)

    @pytest.mark.parametrize("mask_nans", [True, False])
    def test_mask_simple(self, mask_nans):
        mask = xs.regrid.create_mask(
            self.ds,
            variable="tas",
            where_operator=">",
            where_threshold=2,
            mask_nans=mask_nans,
        )
        assert isinstance(mask, xr.DataArray)
        assert mask.attrs["where_threshold"] == "tas > 2"
        assert mask.attrs["mask_NaNs"] == str(mask_nans)
        assert "time" not in mask.dims
        np.testing.assert_allclose(mask, np.stack([np.array([0 if mask_nans else 1, 0, 0, 1, 1, 1])] * 4))

        mask2 = xs.regrid.create_mask(self.ds["tas"], where_operator=">", where_threshold=2, mask_nans=mask_nans)
        assert mask2.equals(mask)

    def test_units(self):
        mask = xs.regrid.create_mask(self.ds, variable="tas", where_operator=">=", where_threshold="2 K")
        assert mask.attrs["where_threshold"] == "tas >= 2 K"
        np.testing.assert_allclose(mask, np.stack([np.array([0, 0, 1, 1, 1, 1])] * 4))

        mask2 = xs.regrid.create_mask(self.ds, variable="tas", where_operator=">=", where_threshold="2 C")
        assert mask2.attrs["where_threshold"] == "tas >= 2 C"
        np.testing.assert_allclose(mask2, np.stack([np.array([0, 0, 0, 0, 0, 0])] * 4))

    def test_error(self):
        with pytest.raises(
            ValueError,
            match="'where_operator' and 'where_threshold' must be used together.",
        ):
            xs.regrid.create_mask(self.ds, variable="tas", where_operator=">")
        with pytest.raises(ValueError, match="A variable needs to be specified when passing a Dataset."):
            xs.regrid.create_mask(self.ds)
