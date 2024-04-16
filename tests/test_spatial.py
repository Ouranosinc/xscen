import dask.array
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import xclim as xc
from shapely.geometry import Polygon

import xscen as xs
from xscen.spatial import _estimate_grid_resolution, _load_lon_lat
from xscen.testing import datablock_3d


class TestCreepFill:
    # Create a 3D datablock
    ds = datablock_3d(
        np.tile(np.arange(1, 37).reshape(6, 6), (3, 1, 1)),
        "tas",
        "lon",
        -70,
        "lat",
        45,
        1,
        1,
        "2000-01-01",
        as_dataset=True,
    )
    ds["mask"] = ds["tas"].isel(time=0) > 0
    # Place a few False values in the mask
    ds["mask"][0, 0] = False
    ds["mask"][3, 3] = False

    @pytest.mark.parametrize(
        ("n", "mode"),
        [(1, "clip"), (2, "clip"), (3, "clip"), (1, "wrap"), (2, "wrap"), (3, "wrap")],
    )
    def test_n(self, n, mode):
        w = xs.spatial.creep_weights(self.ds["mask"], n=n, mode=mode)
        out = xs.spatial.creep_fill(self.ds["tas"], w)

        if mode == "clip":
            neighbours_0 = {
                1: [2, 7, 8],
                2: [2, 7, 8, 3, 9, 13, 14, 15],
                3: [
                    2,
                    7,
                    8,
                    3,
                    9,
                    13,
                    14,
                    15,
                    4,
                    10,
                    16,
                    19,
                    20,
                    21,
                ],  # 22 is False, thus not included
            }
            neighbours_3 = {
                # For these n, the average is the same as the original value
                1: [22],
                2: [22],
                # Here all the values are included, except the False ones
                3: [
                    (self.ds["tas"].isel(time=0).sum().values - 22 - 1)
                    / (self.ds["mask"].count().values - 2)
                ],
            }
        else:
            neighbours_0 = {
                1: [36, 31, 32, 6, 2, 12, 7, 8],
                2: [
                    29,
                    30,
                    25,
                    26,
                    27,
                    35,
                    36,
                    31,
                    32,
                    33,
                    5,
                    6,
                    2,
                    3,
                    11,
                    12,
                    7,
                    8,
                    9,
                    17,
                    18,
                    13,
                    14,
                    15,
                ],
                3: [
                    (
                        np.sum(np.arange(1, 37))
                        - 22
                        - 1
                        + np.sum([19, 20, 21, 23, 24])
                        + np.sum([4, 10, 16, 28, 34])
                    )
                    / (36 - 2 + 10)
                ],
            }
            neighbours_3 = {
                # For these n, the average is the same as the original value
                1: [22],
                2: [22],
                # Here all the values are included, except the False ones
                3: [
                    (
                        np.sum(np.arange(1, 37))
                        - 22
                        - 1
                        + np.sum(np.arange(2, 7))
                        + np.sum([7, 13, 19, 25, 31])
                    )
                    / (36 - 2 + 10)
                ],
            }

        np.testing.assert_allclose(
            out.isel(lat=0, lon=0), np.tile(np.mean(neighbours_0[n]), 3)
        )
        np.testing.assert_allclose(
            out.isel(lat=3, lon=3), np.tile(np.mean(neighbours_3[n]), 3)
        )

    def test_wrong_mode(self):
        with pytest.raises(ValueError, match="mode must be either"):
            xs.spatial.creep_weights(self.ds["mask"], n=1, mode="wrong")

    def test_n0(self):
        w = xs.spatial.creep_weights(self.ds["mask"], n=0, mode="clip")
        out = xs.spatial.creep_fill(self.ds["tas"], w)
        np.testing.assert_equal(out.isel(lat=0, lon=0), np.tile(np.nan, 3))
        np.testing.assert_equal(out.isel(lat=3, lon=3), np.tile(np.nan, 3))


class TestSubset:
    ds = datablock_3d(
        np.ones((3, 50, 50)),
        "tas",
        "lon",
        -70,
        "lat",
        45,
        1,
        1,
        "2000-01-01",
        as_dataset=True,
    )

    @pytest.mark.parametrize(
        ("kwargs", "name"),
        [
            ({"lon": -70, "lat": 45}, None),
            ({"lon": [-53.3, -69.6], "lat": [49.3, 46.6]}, "foo"),
        ],
    )
    def test_subset_gridpoint(self, kwargs, name):
        with pytest.warns(UserWarning, match="tile_buffer is not used"):
            out = xs.spatial.subset(
                self.ds, "gridpoint", name=name, tile_buffer=5, **kwargs
            )

        if isinstance(kwargs["lon"], list):
            expected = {
                "lon": [np.round(k) for k in kwargs["lon"]],
                "lat": [np.round(k) for k in kwargs["lat"]],
            }
        else:
            expected = {
                "lon": [np.round(kwargs["lon"])],
                "lat": [np.round(kwargs["lat"])],
            }

        assert (
            f"gridpoint spatial subsetting on {len(expected['lon'])} coordinates"
            in out.attrs["history"]
        )
        np.testing.assert_array_equal(out["lon"], expected["lon"])
        np.testing.assert_array_equal(out["lat"], expected["lat"])
        if name:
            assert out.attrs["cat:domain"] == name
        else:
            assert "cat:domain" not in out.attrs

    @pytest.mark.parametrize(
        ("kwargs", "tile_buffer", "method"),
        [
            ({"lon_bnds": [-63, -60], "lat_bnds": [47, 50]}, 0, "bbox"),
            ({"lon_bnds": [-63, -60], "lat_bnds": [47, 50]}, 5, "bbox"),
            ({}, 0, "shape"),
            ({}, 5, "shape"),
            ({"buffer": 3}, 5, "shape"),
        ],
    )
    def test_subset_bboxshape(self, kwargs, tile_buffer, method):
        if method == "shape":
            gdf = gpd.GeoDataFrame(
                {"geometry": [Polygon([(-63, 47), (-63, 50), (-60, 50), (-60, 47)])]}
            )
            kwargs["shape"] = gdf

        if "buffer" in kwargs:
            with pytest.raises(
                ValueError, match="Both tile_buffer and clisops' buffer were requested."
            ):
                xs.spatial.subset(self.ds, method, tile_buffer=tile_buffer, **kwargs)
        else:
            out = xs.spatial.subset(self.ds, method, tile_buffer=tile_buffer, **kwargs)

            assert f"{method} spatial subsetting" in out.attrs["history"]
            if tile_buffer:
                assert f"with buffer={tile_buffer}" in out.attrs["history"]
                np.testing.assert_array_equal(
                    out["lon"],
                    np.arange(
                        np.max([-63 - tile_buffer, self.ds.lon.min()]),
                        np.min([-60 + tile_buffer + 1, self.ds.lon.max()]),
                    ),
                )
                np.testing.assert_array_equal(
                    out["lat"],
                    np.arange(
                        np.max([47 - tile_buffer, self.ds.lat.min()]),
                        np.min([50 + tile_buffer + 1, self.ds.lat.max()]),
                    ),
                )

            else:
                assert "with no buffer" in out.attrs["history"]
                np.testing.assert_array_equal(out["lon"], np.arange(-63, -59))
                np.testing.assert_array_equal(out["lat"], np.arange(47, 51))

    def test_subset_sel(self):
        ds = datablock_3d(
            np.ones((3, 50, 50)),
            "tas",
            "rlon",
            -10,
            "rlat",
            0,
            1,
            1,
            "2000-01-01",
            as_dataset=True,
        )

        with pytest.raises(KeyError):
            xs.spatial.subset(ds, "sel", lon=[-75, -70], lat=[-5, 0])
        out = xs.spatial.subset(ds, "sel", rlon=[-5, 5], rlat=[0, 3])

        assert "sel subsetting" in out.attrs["history"]
        np.testing.assert_array_equal(out["rlon"], np.arange(-5, 6))
        np.testing.assert_array_equal(out["rlat"], np.arange(0, 4))

    def test_history(self):
        ds = self.ds.copy(deep=True)
        ds.attrs["history"] = "this is previous history"
        out = xs.spatial.subset(ds, "gridpoint", lon=-70, lat=45)

        assert "this is previous history" in out.attrs["history"].split("\n")[1]
        assert "gridpoint spatial subsetting" in out.attrs["history"].split("\n")[0]

    def test_subset_wrong_method(self):
        with pytest.raises(ValueError, match="Subsetting type not recognized"):
            xs.spatial.subset(self.ds, "wrong", lon=-70, lat=45)

    def test_subset_no_attributes(self):
        ds = self.ds.copy()
        ds.lat.attrs = {}
        ds.lon.attrs = {}
        assert "latitude" not in ds.cf

        xs.spatial.subset(
            ds,
            "bbox",
            name="test",
            lon_bnds=[-63, -60],
            lat_bnds=[47, 50],
        )


def test_dask_coords():
    ds = datablock_3d(
        np.ones((3, 50, 50)),
        "tas",
        "rlon",
        -10,
        "rlat",
        0,
        1,
        1,
        "2000-01-01",
        as_dataset=True,
    )
    # Transform the coordinates to dask arrays
    lon_attrs = ds["lon"].attrs
    ds["lon"] = xr.DataArray(
        dask.array.from_array(ds["lon"].data, chunks=(1, 1)),
        dims=ds["lon"].dims,
        attrs=lon_attrs,
    )
    lat_attrs = ds["lat"].attrs
    ds["lat"] = xr.DataArray(
        dask.array.from_array(ds["lat"].data, chunks=(1, 1)),
        dims=ds["lat"].dims,
        attrs=lat_attrs,
    )
    assert xc.core.utils.uses_dask(ds.cf["longitude"])

    ds = _load_lon_lat(ds)
    assert not xc.core.utils.uses_dask(ds.cf["longitude"])
    assert not xc.core.utils.uses_dask(ds.cf["latitude"])


@pytest.mark.parametrize(("lon_res", "lat_res"), [(0.5, 1), (1, 0.5)])
def test_estimate_res_1d(lon_res, lat_res):
    ds = datablock_3d(
        np.ones((3, 5, 5)),
        "tas",
        "lon",
        -10,
        "lat",
        0,
        lon_res,
        lat_res,
        "2000-01-01",
        as_dataset=True,
    )
    lon_res_est, lat_res_est = _estimate_grid_resolution(ds)
    assert lon_res_est == lon_res
    assert lat_res_est == lat_res


@pytest.mark.parametrize(("lon_res", "lat_res"), [(0.5, 1), (1, 0.5)])
def test_estimate_res_2d(lon_res, lat_res):
    ds = datablock_3d(
        np.ones((3, 5, 5)),
        "tas",
        "rlon",
        -10,
        "rlat",
        0,
        lon_res,
        lat_res,
        "2000-01-01",
        as_dataset=True,
    )
    lon_res_est, lat_res_est = _estimate_grid_resolution(ds)
    np.testing.assert_allclose(lon_res_est, ds.lon.diff("rlon").max())
    np.testing.assert_allclose(lat_res_est, ds.lat.diff("rlat").max())
