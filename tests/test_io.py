import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import xclim as xc
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs
from xscen.testing import datablock_3d


@pytest.mark.parametrize("suffix", [".zarr", ".zarr.zip", "h5", "nc"])
def test_get_engine(tmpdir, suffix):
    if suffix in [".zarr", ".zarr.zip"]:
        path = "some/path" + suffix
        assert xs.io.get_engine(path) == "zarr"
    else:
        ds = timeseries(
            np.zeros(60),
            variable="tas",
            as_dataset=True,
        )
        ds.to_netcdf(
            Path(tmpdir) / f"test.nc",
            engine="netcdf4" if suffix == "nc" else "h5netcdf",
        )
        assert xs.io.get_engine(Path(tmpdir) / f"test.nc") in [
            "netcdf4",
            "h5netcdf",
        ]  # Hard to predict which one


class TestEstimateChunks:
    ds = datablock_3d(
        np.zeros((50, 100, 150)),
        variable="tas",
        x="lon",
        x_start=-70,
        x_step=0.1,
        y="lat",
        y_start=45,
        y_step=-0.1,
        as_dataset=True,
    )
    ds2 = ds.copy()
    ds2["tas"] = ds2["tas"].astype(np.float32)

    def test_normal(self):
        out1 = xs.io.estimate_chunks(self.ds, dims=["time", "lat", "lon"], target_mb=1)
        assert out1 == {"time": 30, "lat": 55, "lon": 85}
        out2 = xs.io.estimate_chunks(self.ds2, dims=["time", "lat", "lon"], target_mb=1)
        assert out2 == {"time": 35, "lat": 70, "lon": 105}
        out3 = xs.io.estimate_chunks(self.ds, dims=["lat", "lon"], target_mb=1)
        assert out3 == {"lon": 65, "lat": 40, "time": -1}
        out4 = xs.io.estimate_chunks(self.ds2, dims=["time"], target_mb=1)
        assert out4 == {"time": 15, "lat": -1, "lon": -1}

    @pytest.mark.parametrize("chunk_per_variable", [True, False])
    @pytest.mark.parametrize("as_file", [True, False])
    def test_multiple_vars(self, tmpdir, chunk_per_variable, as_file):
        ds = self.ds.copy()
        ds["pr"] = ds["tas"].isel(time=0)

        if as_file:
            ds.to_netcdf(Path(tmpdir) / "test.nc")
            ds = Path(tmpdir) / "test.nc"

        out = xs.io.estimate_chunks(
            ds, dims=["lat", "lon"], target_mb=1, chunk_per_variable=chunk_per_variable
        )
        if chunk_per_variable is False:
            assert out == {"lon": 65, "lat": 40, "time": -1}
        else:
            assert out == {
                "tas": {"lon": 65, "lat": 40, "time": -1},
                "pr": {"lon": 150, "lat": 100},
            }


class TestSubsetMaxsize:
    def test_normal(self):
        ds = datablock_3d(
            np.zeros((1500, 5, 5)),
            variable="tas",
            x="lon",
            x_start=-70,
            x_step=0.1,
            y="lat",
            y_start=45,
            y_step=-0.1,
            as_dataset=True,
        )
        ds["pr"] = ds["tas"]
        # First, test with a dataset that is already small enough
        out = xs.io.subset_maxsize(ds, maxsize_gb=1)
        assert len(out) == 1
        assert out[0].equals(ds)

        out = xs.io.subset_maxsize(ds, maxsize_gb=0.0005)
        assert len(out) == 2
        assert xr.concat(out, dim="time").equals(ds)

    def test_error(self):
        ds = datablock_3d(
            np.zeros((1, 50, 10)),
            variable="tas",
            x="lon",
            x_start=-70,
            x_step=0.1,
            y="lat",
            y_start=45,
            y_step=-0.1,
            as_dataset=True,
        )
        ds = ds.isel(time=0)

        with pytest.raises(NotImplementedError, match="does not contain a"):
            xs.io.subset_maxsize(ds, maxsize_gb=1e-15)


def test_clean_incomplete(tmpdir):
    ds = datablock_3d(
        np.ones((5, 5, 5)),
        variable="tas",
        x="lon",
        x_start=-70,
        x_step=0.1,
        y="lat",
        y_start=45,
        y_step=-0.1,
        as_dataset=True,
    )
    ds["pr"] = ds["tas"].copy()
    ds.to_zarr(Path(tmpdir) / "test.zarr")

    xs.io.clean_incomplete(Path(tmpdir) / "test.zarr", complete=["tas"])
    assert Path.exists(Path(tmpdir) / "test.zarr/tas")
    assert not Path.exists(Path(tmpdir) / "test.zarr/pr")

    ds2 = xr.open_zarr(Path(tmpdir) / "test.zarr")
    assert "pr" not in ds2
    assert ds2.equals(ds[["tas"]])


class TestRechunkForSaving:
    @pytest.mark.parametrize(
        "dims, xy",
        [
            (["lon", "lat"], True),
            (["lon", "lat"], False),
            (["rlon", "rlat"], True),
            (["rlon", "rlat"], False),
        ],
    )
    def test_options(self, datablock_3d, dims, xy):
        ds = datablock_3d(
            np.random.random((30, 30, 50)),
            variable="tas",
            x=dims[0],
            x_start=-70 if dims[0] == "lon" else 0,
            y=dims[1],
            y_start=45 if dims[1] == "lat" else 0,
            as_dataset=True,
        )
        x = "X" if xy else dims[0]
        y = "Y" if xy else dims[1]
        new_chunks = {y: 10, x: 10, "time": 20}
        ds_ch = xs.io.rechunk_for_saving(ds, new_chunks)
        for dim, chunks in ds_ch.chunks.items():
            dim = (
                dim
                if (xy is False or dim == "time")
                else "X" if dim == dims[0] else "Y"
            )
            assert chunks[0] == new_chunks[dim]

    def test_variables(self, datablock_3d):
        ds = datablock_3d(
            np.random.random((30, 30, 50)),
            variable="tas",
            x="lon",
            x_start=-70,
            y="lat",
            y_start=45,
            as_dataset=True,
        )
        ds["pr"] = datablock_3d(
            np.random.random((30, 30, 50)),
            variable="pr",
            x="lon",
            x_start=-70,
            y="lat",
            y_start=45,
            as_dataset=False,
        )

        new_chunks = {
            "tas": {"time": 20, "lon": 10, "lat": 7},
            "pr": {"time": 10, "lon": 5, "lat": 3},
        }
        ds_ch = xs.io.rechunk_for_saving(ds, new_chunks)
        for v in ds_ch.data_vars:
            for dim, chunks in zip(list(ds.dims), ds_ch[v].chunks):
                assert chunks[0] == new_chunks[v][dim]


class TestToTable:
    ds = xs.utils.unstack_dates(
        xr.merge(
            [
                xs.testing.datablock_3d(
                    np.random.random_sample((20, 3, 2)),
                    v,
                    "lon",
                    0,
                    "lat",
                    0,
                    1,
                    1,
                    "1993-01-01",
                    "QS-JAN",
                )
                for v in ["tas", "pr", "snw"]
            ]
        )
        .stack(site=["lat", "lon"])
        .reset_index("site")
        .assign_coords(site=list("abcdef"))
    ).transpose("season", "time", "site")
    ds.attrs = {"foo": "bar", "baz": 1, "qux": 2.0}

    @pytest.mark.parametrize(
        "multiple, as_dataset", [(True, True), (False, True), (False, False)]
    )
    def test_normal(self, multiple, as_dataset):
        if multiple is False:
            if as_dataset:
                ds = self.ds[["tas"]].copy()
            else:
                ds = self.ds["tas"].copy()
        else:
            ds = self.ds.copy()

        # Default
        tab = xs.io.to_table(ds)
        assert tab.shape == (120, 5 if multiple else 3)  # 3 vars + 2 aux coords
        assert tab.columns.names == ["variable"] if multiple else [None]
        assert tab.index.names == ["season", "time", "site"]
        # Season order is chronological, rather than alphabetical
        np.testing.assert_array_equal(
            tab.xs("1993", level="time")
            .xs("a", level="site")
            .index.get_level_values("season"),
            ["JFM", "AMJ", "JAS", "OND"],
        )

        if multiple:
            # Variable in the index, thus no coords
            tab = xs.io.to_table(
                ds, row=["time", "variable"], column=["season", "site"], coords=False
            )
            assert tab.shape == (15, 24)
            assert tab.columns.names == ["season", "site"]
            np.testing.assert_array_equal(
                tab.loc[("1993", "pr"), ("JFM",)], ds.pr.sel(time="1993", season="JFM")
            )
            # Ensure that the coords are not present
            assert (
                len(
                    set(tab.index.get_level_values("variable").unique()).difference(
                        ["tas", "pr", "snw"]
                    )
                )
                == 0
            )

    def test_sheet(self):
        tab = xs.io.to_table(
            self.ds,
            row=["time", "variable"],
            column=["season"],
            sheet="site",
            coords=False,
        )
        assert set(tab.keys()) == {("a",), ("b",), ("c",), ("d",), ("e",), ("f",)}
        assert tab[("a",)].shape == (15, 4)  # 5 time * 3 variable X 4 season

    def test_error(self):
        with pytest.raises(ValueError, match="Repeated dimension names."):
            xs.io.to_table(
                self.ds, row=["time", "variable"], column=["season", "site", "time"]
            )
        with pytest.raises(ValueError, match="Passed row, column and sheet"):
            xs.io.to_table(
                self.ds, row=["time", "variable"], column=["season", "site", "foo"]
            )
        with pytest.raises(
            NotImplementedError,
            match="Keeping auxiliary coords is not implemented when",
        ):
            xs.io.to_table(
                self.ds,
                row=["time", "variable"],
                column=["season", "site"],
                coords=True,
            )

    @pytest.mark.parametrize("as_dataset", [True, False])
    def test_make_toc(self, as_dataset):
        ds = self.ds.copy()
        for v in ds.data_vars:
            ds[v].attrs["long_name"] = f"Long name for {v}"
            ds[v].attrs["long_name_fr"] = f"Nom long pour {v}"

        if as_dataset is False:
            ds = ds["tas"]

        with xc.set_options(metadata_locales="fr"):
            toc = xs.io.make_toc(ds)

        if as_dataset:
            assert toc.shape == (8, 2)
            assert toc.columns.tolist() == ["Description", "Unités"]
            assert toc.index.tolist() == [
                "tas",
                "pr",
                "snw",
                "",
                "Attributs globaux",
                "foo",
                "baz",
                "qux",
            ]
            assert toc.loc["tas", "Description"] == "Nom long pour tas"
            assert toc.loc["tas", "Unités"] == "K"
        else:
            assert toc.shape == (1, 2)
            assert toc.columns.tolist() == ["Description", "Unités"]
            assert toc.index.tolist() == ["tas"]
            assert toc.loc["tas", "Description"] == "Nom long pour tas"
            assert toc.loc["tas", "Unités"] == "K"


def test_round_bits():
    da = datablock_3d(
        np.random.random((30, 30, 50)),
        variable="tas",
        x="lon",
        x_start=-70,
        y="lat",
        y_start=45,
    )
    dar = xs.io.round_bits(da, 12)
    # Close but NOT equal, meaning something happened
    np.testing.assert_allclose(da, dar, rtol=0.013)
    # There's always a chance of having a randomly chosen number with only zeros in the bit rounded part of the mantissa
    # Assuming a uniform distribution of binary numbers (which it is not), the chance of this happening should be:
    # 2^(23 - 12 + 1) / 2^24 = 2^(-12) ~ 0.02 % (but we'll allow 1% of values to be safe)
    assert (da != dar).sum() > (0.99 * da.size)


class TestSaveToZarr:
    @pytest.mark.parametrize(
        "vname,vtype,bitr,exp",
        [
            ("tas", np.float32, 12, 12),
            ("tas", np.float32, False, None),
            ("tas", np.int32, 12, None),
            ("tas", np.int32, {"tas": 2}, "error"),
            ("tas", object, {"pr": 2}, None),
            ("tas", np.float64, True, 12),
        ],
    )
    def test_guess_bitround(self, vname, vtype, bitr, exp):
        if exp == "error":
            with pytest.raises(ValueError):
                xs.io._get_keepbits(bitr, vname, vtype)
        else:
            assert xs.io._get_keepbits(bitr, vname, vtype) == exp


class TestSaveToNetcdf:
    def test_normal(self, tmpdir):
        ds = datablock_3d(
            np.tile(np.arange(1111, 1121), 15).reshape(15, 5, 2) * 1e-7,
            variable="tas",
            x="lon",
            x_start=-70,
            y="lat",
            y_start=45,
            as_dataset=True,
        )
        ds["pr"] = ds["tas"].copy()
        ds["other"] = ds["tas"].copy()

        xs.save_to_netcdf(
            ds,
            Path(tmpdir) / "test.nc",
            rechunk={"time": 5, "lon": 2, "lat": 2},
            bitround={"tas": 2, "pr": 3},
        )

        ds2 = xr.open_dataset(Path(tmpdir) / "test.nc", chunks={})
        assert ds2.tas.chunks == ((5, 5, 5), (2, 2, 1), (2,))

        np.testing.assert_array_almost_equal(
            ds2.tas.isel(time=0, lat=0, lon=0), [0.00010681], decimal=8
        )
        assert ds2.tas.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] == 2
        np.testing.assert_array_almost_equal(
            ds2.pr.isel(time=0, lat=0, lon=0), [0.00011444], decimal=8
        )
        assert ds2.pr.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] == 3
        np.testing.assert_array_almost_equal(
            ds2.other.isel(time=0, lat=0, lon=0), [0.0001111], decimal=8
        )
        assert ds2.other.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] == 12
