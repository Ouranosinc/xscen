import numpy as np
import pytest
import xarray as xr
import xclim as xc

import xscen as xs


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


def test_round_bits(datablock_3d):
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
