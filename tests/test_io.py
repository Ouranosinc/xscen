import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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
            Path(tmpdir) / "test.nc",
            engine="netcdf4" if suffix == "nc" else "h5netcdf",
        )
        assert xs.io.get_engine(Path(tmpdir) / "test.nc") in [
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
    ds["just_a_variable"] = xr.DataArray(np.zeros(50), dims="new_dim")

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

        out = xs.io.estimate_chunks(ds, dims=["lat", "lon"], target_mb=1, chunk_per_variable=chunk_per_variable)
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


class TestCleanIncomplete:
    @pytest.mark.parametrize("which", ["complete", "incomplete"])
    def test_complete(self, tmpdir, which):
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

        if which == "complete":
            xs.io.clean_incomplete(Path(tmpdir) / "test.zarr", complete=["tas"])
        else:
            xs.io.clean_incomplete(Path(tmpdir) / "test.zarr", incomplete=["pr"])
        assert (Path(tmpdir) / "test.zarr/tas").exists()
        assert not (Path(tmpdir) / "test.zarr/pr").exists()
        assert (Path(tmpdir) / "test.zarr/.zmetadata").exists()

        ds2 = xr.open_zarr(Path(tmpdir) / "test.zarr")
        assert "pr" not in ds2
        assert ds2.equals(ds[["tas"]])

    def test_error(self, tmpdir):
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

        with pytest.raises(ValueError, match="Use either"):
            xs.io.clean_incomplete(Path(tmpdir) / "test.zarr", complete=["tas"], incomplete=["pr"])


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
    def test_options(self, dims, xy):
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
            dim = dim if (xy is False or dim == "time") else "X" if dim == dims[0] else "Y"
            assert chunks[0] == new_chunks[dim]

    def test_variables(self):
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
            for dim, chunks in zip(list(ds.dims), ds_ch[v].chunks, strict=False):
                assert chunks[0] == new_chunks[v][dim]


class TestToTable:
    ds = xs.utils.unstack_dates(
        xr.merge(
            [
                xs.testing.datablock_3d(
                    np.ones((20, 3, 2)),
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

    @pytest.mark.parametrize("multiple, as_dataset", [(True, True), (False, True), (False, False)])
    def test_normal(self, tmpdir, multiple, as_dataset):
        if multiple is False:
            if as_dataset:
                ds = self.ds[["tas"]].copy()
            else:
                ds = self.ds["tas"].copy()
        else:
            ds = self.ds.copy()

        # Default
        xs.save_to_table(ds, Path(tmpdir) / "test.csv")
        saved = pd.read_csv(Path(tmpdir) / "test.csv")
        tab = xs.io.to_table(ds)

        assert tab.shape == (
            120,
            5 if multiple else 3,
        )  # 3 variables + 2 coords that are not dimensions
        assert saved.shape == (
            120,
            8 if multiple else 6,
        )  # everything gets mapped, so dimensions are included in the columns
        assert tab.columns.names == ["variable"] if multiple else [None]
        assert set(saved.columns) == {"season", "time", "site", "lat", "lon", "pr", "snw", "tas"} if multiple else {"season", "time", "site", "tas"}
        assert tab.index.names == ["season", "time", "site"]
        # Season order is chronological, rather than alphabetical
        np.testing.assert_array_equal(
            tab.xs("1993", level="time").xs("a", level="site").index.get_level_values("season"),
            ["JFM", "AMJ", "JAS", "OND"],
        )
        np.testing.assert_array_equal(saved.loc[0, "season"], "JFM")

        if multiple:
            # Variable in the index, thus no coords
            xs.save_to_table(
                ds,
                Path(tmpdir) / "test.xlsx",
                row=["time", "variable"],
                column=["season", "site"],
                coords=False,
            )
            tab = xs.io.to_table(ds, row=["time", "variable"], column=["season", "site"], coords=False)
            saved = pd.read_excel(Path(tmpdir) / "test.xlsx")

            assert tab.shape == (15, 24)
            assert saved.shape == (17, 26)  # Because of the headers
            assert tab.columns.names == ["season", "site"]
            np.testing.assert_array_equal(tab.loc[("1993", "pr"), ("JFM",)], ds.pr.sel(time="1993", season="JFM"))
            # Ensure that the coords are not present
            assert len(set(tab.index.get_level_values("variable").unique()).difference(["tas", "pr", "snw"])) == 0
            # Excel is not the prettiest thing to test
            np.testing.assert_array_equal(saved.iloc[2, 2:], np.tile([1], 24))
            assert saved.iloc[0, 2] == "a"
            assert saved.iloc[2, 0] == datetime.datetime(1993, 1, 1, 0, 0)

    def test_sheet(self, tmpdir):
        xs.save_to_table(
            self.ds,
            Path(tmpdir) / "test.xlsx",
            row=["time", "variable"],
            column=["season"],
            sheet="site",
            coords=False,
        )
        saved = pd.read_excel(Path(tmpdir) / "test.xlsx", sheet_name=["a", "b", "c", "d", "e", "f"])  # This is a test by itself
        tab = xs.io.to_table(
            self.ds,
            row=["time", "variable"],
            column=["season"],
            sheet="site",
            coords=False,
        )

        assert set(tab.keys()) == {("a",), ("b",), ("c",), ("d",), ("e",), ("f",)}
        assert tab[("a",)].shape == (15, 4)  # 5 time * 3 variable X 4 season
        assert saved["a"].shape == (15, 6)  # Because of the headers

    def test_kwargs(self, tmpdir):
        xs.save_to_table(
            self.ds,
            Path(tmpdir) / "test.xlsx",
            row=["time", "variable"],
            column=["season", "site"],
            coords=False,
            datetime_format="dd/mm/yyyy",
        )
        saved = pd.read_excel(Path(tmpdir) / "test.xlsx")
        assert saved.iloc[2, 0] == datetime.datetime(1993, 1, 1, 0, 0)  # No real way to test the format

    def test_multiindex(self, tmpdir):
        xs.save_to_table(
            self.ds,
            Path(tmpdir) / "test.csv",
            row=["time", "variable"],
            column=["season", "site"],
            coords=False,
            row_sep="|",
            col_sep=";",
        )
        out = pd.read_csv(Path(tmpdir) / "test.csv")
        assert out.shape == (15, 25)
        assert out.columns[0] == "time|variable"
        assert out.columns[1] == "JFM;a"

    def test_error(self, tmpdir):
        with pytest.raises(ValueError, match="Repeated dimension names."):
            xs.save_to_table(
                self.ds,
                Path(tmpdir) / "test.xlsx",
                row=["time", "variable"],
                column=["season", "site", "time"],
            )
        with pytest.raises(ValueError, match="Passed row, column and sheet"):
            xs.save_to_table(
                self.ds,
                Path(tmpdir) / "test.xlsx",
                row=["time", "variable"],
                column=["season", "site", "foo"],
            )
        with pytest.raises(
            NotImplementedError,
            match="Keeping auxiliary coords is not implemented when",
        ):
            xs.save_to_table(
                self.ds,
                Path(tmpdir) / "test.xlsx",
                row=["time", "variable"],
                column=["season", "site"],
                coords=True,
            )
        with pytest.raises(ValueError, match="Output format could not be inferred"):
            xs.save_to_table(self.ds, Path(tmpdir) / "test")
        with pytest.raises(ValueError, match="is only valid with excel as the output format"):
            xs.save_to_table(self.ds, Path(tmpdir) / "test.csv", sheet="site")
        with pytest.raises(ValueError, match="but the output format is not Excel."):
            xs.save_to_table(self.ds, Path(tmpdir) / "test.csv", add_toc=True)

    @pytest.mark.parametrize("as_dataset", [True, False])
    def test_make_toc(self, tmpdir, as_dataset):
        ds = self.ds.copy()
        for v in ds.data_vars:
            ds[v].attrs["long_name"] = f"Long name for {v}"
            ds[v].attrs["long_name_fr"] = f"Nom long pour {v}"

        if as_dataset is False:
            ds = ds["tas"]

        with xc.set_options(metadata_locales="fr"):
            xs.save_to_table(ds, Path(tmpdir) / "test.xlsx", add_toc=True)

        toc = pd.read_excel(Path(tmpdir) / "test.xlsx", sheet_name="Contenu")
        toc = toc.set_index("Unnamed: 0" if as_dataset else "Variable")

        if as_dataset:
            assert toc.shape == (8, 2)
            assert toc.columns.tolist() == ["Description", "Unités"]
            assert toc.index.tolist() == [
                "tas",
                "pr",
                "snw",
                np.nan,
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

    @pytest.mark.parametrize("mode", ["f", "o", "a"])
    @pytest.mark.parametrize("itervar", [True, False])
    def test_mode(self, tmpdir, mode, itervar):
        ds1 = timeseries(
            np.arange(1, 5),
            variable="tas",
            as_dataset=True,
        )
        xs.save_to_zarr(ds1, Path(tmpdir) / "test.zarr")

        ds2 = timeseries(
            np.arange(10, 14),
            variable="tas",
            as_dataset=True,
        )
        ds2["pr"] = ds2["tas"].copy()
        ds2 = ds2[["pr", "tas"]]

        if mode == "f":
            with pytest.raises(ValueError, match="exists in dataset"):
                xs.save_to_zarr(ds2, Path(tmpdir) / "test.zarr", mode=mode, itervar=itervar)
            assert not (Path(tmpdir) / "test.zarr/pr").exists()
            if itervar:
                # Essentially just to reach 100% coverage and make sure the function doesn't crash with mode="f" and itervar=True
                xs.save_to_zarr(ds2, Path(tmpdir) / "test2.zarr", mode=mode, itervar=itervar)
                ds3 = xr.open_zarr(Path(tmpdir) / "test2.zarr")
                np.testing.assert_array_almost_equal(ds3.tas.isel(time=0), [10])
                np.testing.assert_array_almost_equal(ds3.pr.isel(time=0), [10])

        elif mode == "o":
            xs.save_to_zarr(ds2, Path(tmpdir) / "test.zarr", mode=mode, itervar=itervar)
            ds3 = xr.open_zarr(Path(tmpdir) / "test.zarr")
            np.testing.assert_array_almost_equal(ds3.tas.isel(time=0), [10])
            np.testing.assert_array_almost_equal(ds3.pr.isel(time=0), [10])

        elif mode == "a":
            # First, try only with variables that are already in the dataset
            xs.save_to_zarr(ds2[["tas"]], Path(tmpdir) / "test.zarr", mode=mode, itervar=itervar)
            ds3 = xr.open_zarr(Path(tmpdir) / "test.zarr")
            np.testing.assert_array_almost_equal(ds3.tas.isel(time=0), [1])

            # Now, try with a new variable
            xs.save_to_zarr(ds2, Path(tmpdir) / "test.zarr", mode=mode, itervar=itervar)
            ds3 = xr.open_zarr(Path(tmpdir) / "test.zarr")
            np.testing.assert_array_almost_equal(ds3.tas.isel(time=0), [1])
            np.testing.assert_array_almost_equal(ds3.pr.isel(time=0), [10])

    @pytest.mark.parametrize("append", [True, False])
    def test_append(self, tmpdir, append):
        ds1 = datablock_3d(
            np.array([[[1, 2], [3, 4]]]),
            variable="tas",
            x="lon",
            x_start=-70,
            y="lat",
            y_start=45,
            as_dataset=True,
        )
        ds2 = datablock_3d(
            np.array([[[11, 12], [13, 14]]]),
            variable="tas",
            x="lon",
            x_start=-70,
            y="lat",
            y_start=45,
            start="2005-01-01",
            as_dataset=True,
        )
        ds2["pr"] = ds2["tas"].copy()
        xs.save_to_zarr(ds1, Path(tmpdir) / "test.zarr", encoding={"tas": {"dtype": "float32"}})

        encoding = {"tas": {"dtype": "int32"}}  # This should be ignored, as the variable is already in the dataset
        if append:
            with pytest.raises(
                ValueError,
                match="is set in zarr_kwargs, all variables must already exist in the dataset.",
            ):
                xs.save_to_zarr(
                    ds2,
                    Path(tmpdir) / "test.zarr",
                    mode="a",
                    zarr_kwargs={"append_dim": "time"},
                    encoding=encoding,
                )
            xs.save_to_zarr(
                ds2[["tas"]],
                Path(tmpdir) / "test.zarr",
                mode="a",
                zarr_kwargs={"append_dim": "time"},
                encoding=encoding,
            )
            out = xr.open_zarr(Path(tmpdir) / "test.zarr")
            np.testing.assert_array_equal(out.tas, np.array([[[1, 2], [3, 4]], [[11, 12], [13, 14]]]))
        else:
            xs.save_to_zarr(ds2, Path(tmpdir) / "test.zarr", mode="a", encoding=encoding)
            out = xr.open_zarr(Path(tmpdir) / "test.zarr")
            np.testing.assert_array_equal(out.tas, np.array([[[1, 2], [3, 4]]]))
            np.testing.assert_array_equal(out.pr, np.array([[[11, 12], [13, 14]]]))
        assert out.tas.dtype == np.float32

    def test_skip(self, tmpdir):
        ds1 = timeseries(
            np.arange(1, 5),
            variable="tas",
            as_dataset=True,
        )
        ds2 = timeseries(
            np.arange(10, 14),
            variable="tas",
            as_dataset=True,
        )
        xs.save_to_zarr(ds1, Path(tmpdir) / "test.zarr")
        xs.save_to_zarr(ds2, Path(tmpdir) / "test.zarr", mode="a")
        ds3 = xr.open_zarr(Path(tmpdir) / "test.zarr")
        np.testing.assert_array_almost_equal(ds3.tas.isel(time=0), [1])


@pytest.mark.parametrize("engine", ["netcdf", "zarr"])
def test_savefuncs_normal(tmpdir, engine):
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
    ds["other"].encoding = {"dtype": "float32"}
    ds.attrs["foo"] = {"bar": 1}
    ds["pr"].attrs["foo"] = {"bar": 2}

    ds = ds.assign_coords(some_coord=("lat", np.array(["hi", "how", "are", "you", "doing"])))
    ds["some_coord"] = ds["some_coord"].astype(object)
    ds["some_coord"].encoding = {"source": "this is a source"}

    rechunk = {"time": 5, "lon": 2, "lat": 2}
    bitround = {"tas": 2, "pr": 3}
    if engine == "netcdf":
        xs.save_to_netcdf(
            ds,
            Path(tmpdir) / "test.nc",
            rechunk=rechunk,
            bitround=bitround,
        )
        ds2 = xr.open_dataset(Path(tmpdir) / "test.nc", chunks={})
    else:
        xs.save_to_zarr(
            ds,
            Path(tmpdir) / "test.zarr",
            rechunk=rechunk,
            bitround=bitround,
        )
        ds2 = xr.open_zarr(Path(tmpdir) / "test.zarr")

    # Chunks
    assert ds2.tas.chunks == ((5, 5, 5), (2, 2, 1), (2,))

    # Dtype
    assert ds2.tas.dtype == np.float64
    assert ds2.other.dtype == np.float32

    # Bitround
    np.testing.assert_array_almost_equal(ds2.tas.isel(time=0, lat=0, lon=0), [0.00010681], decimal=8)
    assert ds2.tas.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] == 2
    np.testing.assert_array_almost_equal(ds2.pr.isel(time=0, lat=0, lon=0), [0.00011444], decimal=8)
    assert ds2.pr.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] == 3
    np.testing.assert_array_almost_equal(ds2.other.isel(time=0, lat=0, lon=0), [0.0001111], decimal=8)
    assert ds2.other.attrs["_QuantizeBitRoundNumberOfSignificantDigits"] == 12

    # Attributes
    assert ds2.attrs["foo"] == "{'bar': 1}"
    assert ds2.pr.attrs["foo"] == "{'bar': 2}"

    if engine == "netcdf":
        assert ds.some_coord.encoding == {"source": "this is a source"}
    else:
        assert ds.some_coord.encoding == {}


class TestRechunk:
    @pytest.mark.parametrize("engine", ["nc", "zarr"])
    def test_rechunk(self, tmpdir, engine):
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

        if engine == "nc":
            xs.save_to_netcdf(
                ds,
                Path(tmpdir) / "test.nc",
            )
        else:
            xs.save_to_zarr(
                ds,
                Path(tmpdir) / "test.zarr",
            )

        (Path(tmpdir) / "test2.zarr").mkdir()

        xs.io.rechunk(
            Path(tmpdir) / f"test.{engine}",
            Path(tmpdir) / "test2.zarr",
            chunks_over_dim={"time": 5, "lon": 2, "lat": 2},
            overwrite=True,
            worker_mem="1GB",
            temp_store=Path(tmpdir) / "temp",
        )
        xs.io.rechunk(
            Path(tmpdir) / f"test.{engine}",
            Path(tmpdir) / "test3.zarr",
            chunks_over_var={"tas": {"time": 5, "lon": 2, "lat": 2}},
            overwrite=True,
            worker_mem="1GB",
            temp_store=Path(tmpdir) / "temp",
        )

        ds2 = xr.open_zarr(Path(tmpdir) / "test2.zarr")
        ds3 = xr.open_zarr(Path(tmpdir) / "test3.zarr")
        assert ds2.tas.chunks == ((5, 5, 5), (2, 2, 1), (2,))
        assert ds2.pr.chunks == ((5, 5, 5), (2, 2, 1), (2,))
        assert ds3.tas.chunks == ((5, 5, 5), (2, 2, 1), (2,))
        assert ds3.pr.chunks == ((15,), (5,), (2,))

    def test_error(self, tmpdir):
        ds = datablock_3d(
            np.tile(np.arange(1111, 1121), 15).reshape(15, 5, 2) * 1e-7,
            variable="tas",
            x="lon",
            x_start=-70,
            y="lat",
            y_start=45,
            as_dataset=True,
        )
        with pytest.raises(ValueError, match="No chunks given. "):
            xs.io.rechunk(ds, Path(tmpdir) / "test.nc", worker_mem="1GB")


def test_zip_zip(tmpdir):
    ds = datablock_3d(
        np.tile(np.arange(1111, 1121), 15).reshape(15, 5, 2) * 1e-7,
        variable="tas",
        x="lon",
        x_start=-70,
        y="lat",
        y_start=45,
        as_dataset=True,
    )
    xs.save_to_zarr(ds, Path(tmpdir) / "test.zarr")
    xs.io.zip_directory(Path(tmpdir) / "test.zarr", Path(tmpdir) / "test.zarr.zip", delete=True)
    assert not (Path(tmpdir) / "test.zarr").exists()

    with xr.open_zarr(Path(tmpdir) / "test.zarr.zip") as ds2:
        assert ds2.equals(ds)

    xs.io.unzip_directory(Path(tmpdir) / "test.zarr.zip", Path(tmpdir) / "test2.zarr")
    with xr.open_zarr(Path(tmpdir) / "test2.zarr") as ds3:
        assert ds3.equals(ds)


def test_save_load_sparse(tmpdir):
    rng = np.random.default_rng()
    da = datablock_3d(
        rng.integers(0, 10, (50, 50, 1)),
        variable="tas",
        units="°C",
        x="lon",
        x_start=-70,
        y="lat",
        y_start=45,
    )
    da = da.where(da > 2)
    w1 = xs.spatial.creep_weights(da.isel(time=0).notnull())
    xs.io.save_sparse(w1, Path(tmpdir) / "sparse.nc")
    w2 = xs.io.load_sparse(Path(tmpdir) / "sparse.nc")

    xr.testing.assert_identical(w1, w2)
