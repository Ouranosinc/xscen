from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import notebooks
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs


class TestSearchDataCatalogs:
    cat = xs.DataCatalog(notebooks / "samples" / "pangeo-cmip6.json")

    @pytest.mark.parametrize(
        "variables_and_freqs, other_arg",
        [
            ({"tasmin": "D"}, None),
            ({"sftlf": "fx"}, "other"),
            ({"tasmin": "D", "sftlf": "fx"}, "exclusion"),
        ],
    )
    def test_basic(self, variables_and_freqs, other_arg):
        out = xs.search_data_catalogs(
            data_catalogs=self.cat,
            variables_and_freqs=variables_and_freqs,
            other_search_criteria=({"experiment": ["ssp585"]} if other_arg == "other" else None),
            exclusions=({"member": "r2.*", "domain": ["gr2"]} if other_arg == "exclusion" else None),
        )
        assert len(out) == 13 if other_arg is None else 2 if other_arg == "other" else 4

    @pytest.mark.parametrize(
        "periods, coverage_kwargs, matches",
        [
            ([["2020", "2030"], ["2035", "2040"]], None, 5),
            ([["2020-01", "2030-12"], ["2035-01", "2040-12"]], None, 5),
            ([["2020-02", "2030-11"], ["2035", "2040"]], None, 5),
            ([["1900", "2030"], ["2035", "2040"]], None, 0),
            ([["1985-01-01", "2014-12-31"]], {"coverage": 1}, 8),  # 8
            ([["1985-01-01", "2015-01-01"]], {"coverage": 1}, 0),  # 0
            ([["1985-01-01", "2015-01-01"]], None, 8),  # default coverage = 0.99
            ([["1985-01-01", "2015-04-20"]], None, 8),
            ([["1985-01-01", "2015-04-20 16:00:01"]], None, 0),  # 0
            ([["1985-01-01", "2015-04-20 15:59:00"]], None, 8),  # 8
            ([["1985-01-02", "2015-04-20 15:45:00"]], None, 8),  # 8
            (
                [["1985-01-02", "2015-04-20 15:46:00"]],
                None,
                0,
            ),  # ensure uses intersection
            ([["1985-01-01", "2020"]], None, 0),  # 0
            ([["2020", "2130"]], {"coverage": 0.70}, 5),
        ],
    )
    def test_periods(self, periods, coverage_kwargs, matches):
        out = xs.search_data_catalogs(
            data_catalogs=self.cat,
            variables_and_freqs={"tasmin": "D"},
            periods=periods,
            coverage_kwargs=coverage_kwargs,
        )
        assert len(out) == matches

    def test_ids(self):
        out = xs.search_data_catalogs(
            data_catalogs=deepcopy(self.cat),
            variables_and_freqs={"tasmin": "D"},
            id_columns=["source"],
        )
        assert len(out) == 3
        assert len(out["NorESM2-MM"].df) == 5

    @pytest.mark.parametrize("allow_resampling", [True, False])
    def test_allow_resampling(self, allow_resampling):
        out = xs.search_data_catalogs(
            data_catalogs=deepcopy(self.cat),
            variables_and_freqs={"tasmin": "YS"},
            allow_resampling=allow_resampling,
        )
        assert len(out) == (13 if allow_resampling else 0)

    @pytest.mark.parametrize(
        "restrict_warming_level,exp",
        [(True, 5), ({"wl": 2, "ignore_member": True}, 5), ({"wl": 4}, 2)],
    )
    def test_warminglevel(self, restrict_warming_level, exp):
        cat = deepcopy(self.cat)
        new_line = deepcopy(cat.df.iloc[13])
        new_line["experiment"] = "ssp245"
        new_line["id"] = xs.catalog.generate_id(new_line.to_frame().T).iloc[0]
        cat.esmcat._df = pd.concat([cat.df, new_line.to_frame().T], ignore_index=True)

        out = xs.search_data_catalogs(
            data_catalogs=cat,
            variables_and_freqs={"tasmax": "D"},
            restrict_warming_level=restrict_warming_level,
        )
        assert len(out) == exp

    @pytest.mark.parametrize("restrict_resolution", [None, "finest", "coarsest"])
    def test_restrict_resolution(self, restrict_resolution):
        cat = deepcopy(self.cat)
        for i in range(2):
            new_line = deepcopy(cat.df.iloc[0])
            new_line["mip_era"] = "CMIP5"
            new_line["activity"] = "CORDEX"
            new_line["institution"] = "CCCma"
            new_line["driving_model"] = "CanESM2"
            new_line["source"] = "CRCM5"
            new_line["experiment"] = "rcp85"
            new_line["member"] = "r1i1p1"
            new_line["domain"] = "NAM-22" if i == 0 else "NAM-11"
            new_line["frequency"] = "day"
            new_line["xrfreq"] = "D"
            new_line["variable"] = ("tasmin",)
            new_line["id"] = xs.catalog.generate_id(new_line.to_frame().T).iloc[0]

            cat.esmcat._df = pd.concat([cat.df, new_line.to_frame().T], ignore_index=True)

        out = xs.search_data_catalogs(
            data_catalogs=cat,
            variables_and_freqs={"tasmin": "D"},
            other_search_criteria={
                "source": ["GFDL-CM4", "CRCM5"],
                "experiment": ["ssp585", "rcp85"],
            },
            restrict_resolution=restrict_resolution,
        )
        if restrict_resolution is None:
            assert len(out) == 4
        elif restrict_resolution == "finest":
            assert len(out) == 2
            assert any("NAM-11" in x for x in out)
            assert any("_gr1" in x for x in out)
        elif restrict_resolution == "coarsest":
            assert len(out) == 2
            assert any("NAM-22" in x for x in out)
            assert any("_gr2" in x for x in out)

    @pytest.mark.parametrize("restrict_members", [None, {"ordered": 2}])
    def test_restrict_members(self, restrict_members):
        out = xs.search_data_catalogs(
            data_catalogs=self.cat,
            variables_and_freqs={"tasmin": "D"},
            other_search_criteria={
                "source": ["NorESM2-LM"],
                "experiment": ["historical"],
            },
            restrict_members=restrict_members,
        )
        assert len(out) == (3 if restrict_members is None else 2)
        if restrict_members is not None:
            assert all(
                o in out.keys()
                for o in [
                    "CMIP_NCC_NorESM2-LM_historical_r1i1p1f1_gn",
                    "CMIP_NCC_NorESM2-LM_historical_r2i1p1f1_gn",
                ]
            )

        # Make sure that those with fewer members are still returned
        assert (
            len(
                xs.search_data_catalogs(
                    data_catalogs=self.cat,
                    variables_and_freqs={"tasmin": "D"},
                    other_search_criteria={
                        "source": ["GFDL-CM4"],
                        "experiment": ["ssp585"],
                        "domain": "gr1",
                    },
                    restrict_members=restrict_members,
                )
            )
            == 1
        )

    @pytest.mark.parametrize("allow_conversion", [True, False])
    def test_allow_conversion(self, allow_conversion):
        out = xs.search_data_catalogs(
            data_catalogs=self.cat,
            variables_and_freqs={"evspsblpot": "D"},
            other_search_criteria={
                "institution": ["NOAA-GFDL"],
                "experiment": ["ssp585"],
            },
            allow_conversion=allow_conversion,
        )
        assert len(out) == (2 if allow_conversion else 0)
        if allow_conversion:
            assert all(v in out[list(out.keys())[0]].unique("variable") for v in ["tasmin", "tasmax"])
            assert "tas" not in out[list(out.keys())[0]].unique("variable")

    def test_no_match(self):
        out = xs.search_data_catalogs(
            data_catalogs=self.cat,
            variables_and_freqs={"tas": "YS"},
            allow_resampling=False,
        )
        assert isinstance(out, dict)
        assert len(out) == 0
        out = xs.search_data_catalogs(
            data_catalogs=self.cat,
            variables_and_freqs={"tas": "D"},
            other_search_criteria={"experiment": "not_real"},
        )
        assert isinstance(out, dict)
        assert len(out) == 0

    def test_input_types(self, samplecat):
        data_catalogs_2 = notebooks / "samples" / "pangeo-cmip6.json"

        assert (
            xs.search_data_catalogs(
                data_catalogs=[samplecat, data_catalogs_2],
                variables_and_freqs={"tas": "D"},
                other_search_criteria={
                    "experiment": "ssp585",
                    "source": "NorESM.*",
                    "member": "r1i1p1f1",
                },
            ).keys()
            == xs.search_data_catalogs(
                data_catalogs=[samplecat, self.cat],
                variables_and_freqs={"tas": "D"},
                other_search_criteria={
                    "experiment": "ssp585",
                    "source": "NorESM.*",
                    "member": "r1i1p1f1",
                },
            ).keys()
        )

    def test_match_histfut(self):
        out = xs.search_data_catalogs(
            data_catalogs=self.cat,
            variables_and_freqs={"tasmin": "D"},
            other_search_criteria={"experiment": "ssp585", "source": "GFDL-CM4"},
            match_hist_and_fut=True,
        )
        k = list(out.keys())[0]
        assert str(sorted(out[k].unique("date_start"))[0]) == "1985-01-01 00:00:00"
        assert str(sorted(out[k].unique("date_start"))[1]) == "2015-01-01 00:00:00"

    def test_fx(self):
        cat = deepcopy(self.cat)
        new_line = deepcopy(cat.df.iloc[0])
        new_line["id"] = new_line["id"].replace(new_line["experiment"], "another_experiment")
        new_line["experiment"] = "another_experiment"
        cat.esmcat._df = pd.concat([cat.df, new_line.to_frame().T], ignore_index=True)

        with pytest.warns(
            UserWarning,
            match="doesn't have the fixed field sftlf, but it can be acquired from ",
        ):
            out = xs.search_data_catalogs(
                data_catalogs=cat,
                variables_and_freqs={"sftlf": "fx"},
                other_search_criteria={"experiment": "another_experiment"},
            )
        assert len(out) == 1
        k = list(out.keys())[0]
        np.testing.assert_array_equal(
            out[k].df["experiment"],
            "another_experiment",
        )


class TestGetWarmingLevel:
    def test_list(self):
        with pytest.warns(FutureWarning, match="get_warming_level has been deprecated."):
            out = xs.get_warming_level(
                ["CMIP6_CanESM5_ssp126_r1i1p1f1", "CMIP6_CanESM5_ssp245_r1i1p1f1"],
                wl=2,
                window=20,
                return_horizon=False,
            )
        assert isinstance(out, list)
        assert out == ["2026", "2024"]

        period = [int(out[0]) - 9, int(out[0]) + 10]
        out2 = xs.get_warming_level_from_period(
            ["CMIP6_CanESM5_ssp126_r1i1p1f1", "CMIP6_CanESM5_ssp245_r1i1p1f1"],
            period,
        )
        assert isinstance(out2, list)
        np.testing.assert_array_almost_equal(out2, [2.0418553921, 2.1353053921])
        assert xs.get_warming_level_from_period("CMIP6_CanESM5_ssp126_r1i1p1f1", list(np.array(period) - 1)) < 2

    def test_string_with_horizon(self):
        out = xs.get_period_from_warming_level("CMIP6_CanESM5_ssp585_r1i1p1f1", wl=2, window=20, return_central_year=False)
        assert isinstance(out, list)
        assert out == ["2013", "2032"]

        out2 = xs.get_warming_level_from_period("CMIP6_CanESM5_ssp585_r1i1p1f1", out)
        np.testing.assert_array_almost_equal(out2, 2.00547622)

    @pytest.mark.parametrize("attrs", ["global", "regional", "regional_w_institution"])
    def test_ds(self, attrs):
        ds = timeseries(
            np.tile(np.arange(1, 2), 30),
            variable="tas",
            start="2001-01-01",
            freq="YS-JAN",
            as_dataset=True,
        )
        attributes = {
            "global": {
                "cat:mip_era": "CMIP6",
                "cat:source": "CanESM5",
                "cat:experiment": "ssp126",
            },
            "regional": {
                "cat:mip_era": "CMIP6",
                "cat:driving_model": "CanESM5",
                "cat:experiment": "ssp126",
            },
            "regional_w_institution": {
                "cat:mip_era": "CMIP6",
                "cat:driving_institution": "CCCma",
                "cat:driving_model": "CCCma-CanESM5",
                "cat:experiment": "ssp126",
            },
        }

        ds.attrs = attributes[deepcopy(attrs)]
        assert xs.get_period_from_warming_level(ds, wl=2, window=20, ignore_member=True, return_central_year=True) == "2026"
        np.testing.assert_array_almost_equal(
            xs.get_warming_level_from_period(ds, [2026 - 9, 2026 + 10], ignore_member=True),
            2.0418553,
        )

    def test_multiple_levels(self):
        # ensure returns multiple warming periods when multiple levels are requested:
        assert xs.get_period_from_warming_level(
            "CMIP6_CanESM5_ssp126_r1i1p1f1",
            wl=np.array([1.5, 2, 2.5, 3]),
            window=21,
            return_central_year=False,
        ) == [["2004", "2024"], ["2016", "2036"], ["2030", "2050"], [None, None]]

        assert xs.get_period_from_warming_level(
            "CMIP6_CanESM5_ssp126_r1i1p1f1",
            wl=np.array([1.5, 2, 2.5, 3]),
            window=20,
            return_central_year=True,
        ) == ["2013", "2026", "2040", None]

    def test_multiple_matches(self):
        # 55 instances of CanESM2-rcp85 in the CSV, but it should still return a single value
        assert xs.get_period_from_warming_level("CMIP5_CanESM2_rcp85_.*", wl=3.5, window=30, return_central_year=True) == "2059"
        np.testing.assert_array_almost_equal(
            xs.get_warming_level_from_period("CMIP5_CanESM2_rcp85_.*", [2059 - 14, 2059 + 15]),
            3.545244,
        )

    def test_odd_window(self):
        assert xs.get_period_from_warming_level("CMIP6_CanESM5_ssp126_r1i1p1f1", wl=2, window=21, return_central_year=False) == ["2016", "2036"]
        np.testing.assert_array_almost_equal(
            xs.get_warming_level_from_period("CMIP6_CanESM5_ssp126_r1i1p1f1", [2016, 2036]),
            2.019456,
        )

    def test_none(self):
        assert (
            xs.get_period_from_warming_level(
                "CMIP6_CanESM5_ssp585_r1i1p1f1",
                wl=20,
                window=20,
                return_central_year=True,
            )
            is None
        )
        assert (
            xs.get_period_from_warming_level(
                "CMIP6_notreal_ssp585_r1i1p1f1",
                wl=20,
                window=20,
                return_central_year=True,
            )
            is None
        )
        with pytest.raises(ValueError, match="s not fully covered by the provided"):
            xs.get_warming_level_from_period("CMIP6_CanESM5_ssp585_r1i1p1f1", ["2100", "2120"])
        assert xs.get_warming_level_from_period("CMIP6_notreal_ssp585_r1i1p1f1", ["2010", "2020"]) is None

    def test_wrong_types(self):
        with pytest.raises(ValueError):
            xs.get_period_from_warming_level({"this": "is not valid."}, wl=2, window=20, return_central_year=False)
        with pytest.raises(ValueError):
            xs.get_period_from_warming_level(
                "CMIP6_CanESM5_ssp585_r1i1p1f1_toomany_underscores",
                wl=2,
                window=20,
                return_central_year=False,
            )
        with pytest.raises(ValueError):
            xs.get_period_from_warming_level(
                "CMIP6_CanESM5_ssp585_r1i1p1f1",
                wl=2,
                window=3.85,
                return_central_year=False,
            )

    def test_DataArray(self):  # noqa: N802
        reals = xr.DataArray(["CMIP6_CanESM5_ssp126_r1i1p1f1"], dims=("x",), coords={"x": [1]})
        out = xs.get_period_from_warming_level(reals, wl=2, return_central_year=True)
        xr.testing.assert_identical(out, reals.copy(data=["2026"]))

        out2 = xs.get_warming_level_from_period(reals, [2026 - 9, 2026 + 10])
        np.testing.assert_array_almost_equal(out2, [2.0418553921])

        reals = xr.DataArray(
            ["CMIP6_CanESM5_ssp126_r1i1p1f1", "CMIP6_CanESM5_ssp370_r1i1p1f1"],
            dims=("x",),
            coords={"x": [1, 2]},
        )
        out = xs.get_period_from_warming_level(reals, wl=[1, 1.5, 2, 2.5], return_central_year=True)
        assert_out = xr.DataArray(
            [["1999", "2013", "2026", "2040"], ["1999", "2013", "2023", "2033"]],
            dims=("x", "wl"),
            coords=dict(**reals.coords, wl=[1, 1.5, 2, 2.5]),
        )
        xr.testing.assert_identical(out, assert_out)

    def test_DataFrame(self):  # noqa: N802
        reals = pd.DataFrame.from_records(
            [
                {
                    "mip_era": "CMIP6",
                    "source": "CanESM5",
                    "experiment": "ssp126",
                    "member": "r1i1p1f1",
                },
                {
                    "mip_era": "CMIP6",
                    "source": "CanESM5",
                    "experiment": "ssp245",
                    "member": "r1i1p1f1",
                },
            ],
            index=["a", "b"],
        )
        out = xs.get_period_from_warming_level(
            reals,
            wl=2,
            window=20,
            return_central_year=True,
        )
        pd.testing.assert_series_equal(out, pd.Series(["2026", "2024"], index=["a", "b"]))

        out = xs.get_period_from_warming_level(
            reals,
            wl=[2, 2.5],
            window=20,
            return_central_year=True,
        )
        pd.testing.assert_series_equal(
            out,
            pd.Series(
                ["2026", "2040", "2024", "2036"],
                index=[("a", 2), ("a", 2.5), ("b", 2), ("b", 2.5)],
            ),
        )

        out2 = xs.get_warming_level_from_period(reals, [2026 - 9, 2026 + 10])
        np.testing.assert_array_almost_equal(out2, [2.0418553921, 2.1353053921])


class TestSubsetWarmingLevel:
    ds = timeseries(
        np.tile(np.arange(1, 2), 50),
        variable="tas",
        start="2000-01-01",
        freq="YS-JAN",
        as_dataset=True,
    ).assign_attrs(
        {
            "cat:mip_era": "CMIP6",
            "cat:source": "CanESM5",
            "cat:experiment": "ssp585",
            "cat:member": "r1i1p1f1",
        }
    )

    def test_default(self):
        ds_sub = xs.subset_warming_level(self.ds, wl=2)

        np.testing.assert_array_equal(ds_sub.time.dt.year[0], 2013)
        np.testing.assert_array_equal(ds_sub.time.dt.year[-1], 2032)
        np.testing.assert_array_equal(ds_sub.warminglevel[0], "+2Cvs1850-1900")
        assert ds_sub.warminglevel.attrs["baseline"] == "1850-1900"
        assert ds_sub.attrs["cat:processing_level"] == "warminglevel-2vs1850-1900"

    def test_kwargs(self):
        ds_sub = xs.subset_warming_level(
            self.ds,
            wl=1,
            window=25,
            tas_baseline_period=["1981", "2010"],
            to_level="tests",
        )

        np.testing.assert_array_equal(ds_sub.time.dt.year[0], 2009)
        np.testing.assert_array_equal(ds_sub.time.dt.year[-1], 2033)
        np.testing.assert_array_equal(ds_sub.warminglevel[0], "+1Cvs1981-2010")
        assert ds_sub.warminglevel.attrs["baseline"] == "1981-2010"
        assert ds_sub.attrs["cat:processing_level"] == "tests"

    @pytest.mark.parametrize("wl", [3.5, 5, 20, [2, 3.5, 5, 20], [3.5, 5, 20]])
    def test_outofrange(self, wl):
        # 3.5 is only partially covered by ds, 5 is out of range but within the csv, 20 is fully out of range
        if not isinstance(wl, list):
            assert xs.subset_warming_level(self.ds, wl=wl) is None
        else:
            ds = xs.subset_warming_level(self.ds, wl=wl)
            assert ds.warminglevel.size == len(wl)
            if len(wl) == 3:
                np.testing.assert_array_equal(ds.tas.isnull().all(), [True])
            else:
                np.testing.assert_array_equal(ds.tas.isnull().all(dim="time"), [False, True, True, True])

    @pytest.mark.parametrize("asaux", [True, False])
    def test_multireals(self, asaux):
        if asaux:
            ds = self.ds.expand_dims(real=[0, 1, 2]).assign_coords(
                realization=(
                    ("real",),
                    [
                        "CMIP6_CanESM5_ssp126_r1i1p1f1",
                        "CMIP6_CanESM5_ssp245_r1i1p1f1",
                        "fake_faux_falsch_falso",
                    ],
                )
            )
        else:
            ds = self.ds.expand_dims(
                realization=[
                    "CMIP6_CanESM5_ssp126_r1i1p1f1",
                    "CMIP6_CanESM5_ssp245_r1i1p1f1",
                    "fake_faux_falsch_falso",
                ]
            )
        ds_sub = xs.subset_warming_level(
            ds,
            wl=1.5,
            to_level="tests",
        )
        np.testing.assert_array_equal(ds_sub.time.dt.year, np.arange(1000, 1020))
        np.testing.assert_array_equal(ds_sub.warminglevel_bounds[:2].dt.year, [[[2004, 2023]], [[2004, 2023]]])
        assert ds_sub.warminglevel_bounds[2].isnull().all()

    def test_multilevels(self):
        ds_sub = xs.subset_warming_level(
            self.ds,
            wl=[1, 2, 3, 20],
            to_level="tests",
        )
        np.testing.assert_array_equal(
            ds_sub.warminglevel,
            ["+1Cvs1850-1900", "+2Cvs1850-1900", "+3Cvs1850-1900", "+20Cvs1850-1900"],
        )
        np.testing.assert_array_equal(ds_sub.tas.isnull().sum("time"), [20, 0, 20, 20])


class TestResample:
    @pytest.mark.parametrize(
        "infrq,meth,outfrq,exp",
        [
            ["MS", "mean", "QS-DEC", [0.47457627, 3, 6.01086957, 9, 11.96666667]],
            [
                "QS-DEC",
                "mean",
                "YS",
                [1.49041096, 5.49041096, 9.49453552, 13.49041096, 17.49041096],
            ],
            ["MS", "std", "2YS", [6.92009239, 6.91557206]],
            [
                "QS",
                "median",
                "YS",
                [1.516437, 5.516437, 9.516437, 13.51092864, 17.516437],
            ],
        ],
    )
    def test_weighted(self, infrq, meth, outfrq, exp):
        da = timeseries(
            np.arange(48),
            variable="tas",
            start="2001-01-01",
            freq=infrq,
        )
        out = xs.extract.resample(da, outfrq, method=meth)
        np.testing.assert_allclose(out.isel(time=slice(0, 5)), exp)

    def test_weighted_wind(self):
        uas = timeseries(
            np.arange(48),
            variable="uas",
            start="2001-01-01",
            freq="MS",
        )
        vas = timeseries(
            np.arange(48),
            variable="vas",
            start="2001-01-01",
            freq="MS",
        )
        ds = xr.merge([uas, vas])
        out = xs.extract.resample(ds.uas, "YS", method="wind_direction", ds=ds)
        np.testing.assert_allclose(out, [5.5260274, 17.5260274, 29.5260274, 41.5136612])

    def test_missing(self):
        da = timeseries(
            np.arange(48),
            variable="tas",
            start="2001-01-01",
            freq="MS",
        )
        out = xs.extract.resample(da, "QS-DEC", method="mean", missing="drop")
        assert out.size == 15

        out = xs.extract.resample(da, "QS-DEC", method="mean", missing="mask")
        assert out.isel(time=0).isnull().all()

    def test_missing_xclim(self):
        arr = np.arange(48).astype(float)
        arr[0] = np.nan
        arr[40:] = np.nan
        da = timeseries(
            arr,
            variable="tas",
            start="2001-01-01",
            freq="MS",
        )
        out = xs.extract.resample(da, "YS", method="mean", missing={"method": "any"})
        assert out.isel(time=0).isnull().all()

        out = xs.extract.resample(da, "YS", method="mean", missing={"method": "pct", "tolerance": 0.6})
        assert out.isel(time=0).notnull().all()
        assert out.isel(time=-1).isnull().all()

    def test_nofreq(self):
        da = timeseries(
            np.arange(72),
            variable="tas",
            start="2001-01-01",
            freq="h",
        )
        da = xr.concat((da[:50], da[55:]), "time")

        out = xs.extract.resample(da, "D")
        np.testing.assert_allclose(out, [11.5, 35.5, 61.473684])

        with pytest.raises(ValueError, match="Can't perform missing checks"):
            out = xs.extract.resample(da, "D", missing="drop")

        out = xs.extract.resample(da, "D", initial_frequency="h", missing="drop")
        assert out.size == 2

        out = xs.extract.resample(da, "D", initial_frequency="h", missing="mask")
        np.testing.assert_array_equal(out.isnull(), [False, False, True])
