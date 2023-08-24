from copy import deepcopy

import numpy as np
import pytest
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs

from .conftest import notebooks


class TestSearchDataCatalogs:
    small_cat = xs.DataCatalog(notebooks / "samples" / "tutorial-catalog.json")
    big_cat = xs.DataCatalog(notebooks / "samples" / "pangeo-cmip6.json")

    @pytest.mark.parametrize(
        "variables_and_freqs, other_arg",
        [
            ({"tas": "D"}, None),
            ({"sftlf": "fx"}, "other"),
            ({"tas": "D", "sftlf": "fx"}, "exclusion"),
        ],
    )
    def test_basic(self, variables_and_freqs, other_arg):
        out = xs.search_data_catalogs(
            data_catalogs=self.small_cat,
            variables_and_freqs=variables_and_freqs,
            other_search_criteria={"experiment": ["ssp370"]}
            if other_arg == "other"
            else None,
            exclusions={"member": "r2.*"} if other_arg == "exclusion" else None,
        )
        assert len(out) == 5 if other_arg is None else 1 if other_arg == "other" else 4

    @pytest.mark.parametrize(
        "periods, coverage_kwargs",
        [
            ([["2020", "2030"], ["2035", "2040"]], None),
            ([["1900", "2030"], ["2035", "2040"]], None),
            ([["2020", "2080"]], None),
            ([["2020", "2080"]], {"coverage": 0.5}),
        ],
    )
    def test_periods(self, periods, coverage_kwargs):
        out = xs.search_data_catalogs(
            data_catalogs=self.small_cat,
            variables_and_freqs={"tas": "D"},
            periods=periods,
            coverage_kwargs=coverage_kwargs,
        )
        assert len(out) == (
            5
            if ((periods[0] == ["2020", "2030"]) or coverage_kwargs is not None)
            else 0
        )

    def test_ids(self):
        out = xs.search_data_catalogs(
            data_catalogs=deepcopy(self.small_cat),
            variables_and_freqs={"tas": "D"},
            id_columns=["source"],
        )
        assert len(out) == 1
        assert len(out["NorESM2-MM"].df) == 5

        # Missing id
        small_cat = deepcopy(self.small_cat)
        small_cat.esmcat._df.loc[
            small_cat.esmcat._df.id
            == "CMIP6_ScenarioMIP_NCC_NorESM2-MM_ssp126_r1i1p1f1_example-region",
            "id",
        ] = None
        assert (
            "CMIP6_ScenarioMIP_NCC_NorESM2-MM_ssp126_r1i1p1f1_example-region"
            not in small_cat.esmcat._df.id.values
        )
        out = xs.search_data_catalogs(
            data_catalogs=deepcopy(self.small_cat),
            variables_and_freqs={"tas": "D"},
        )
        assert len(out) == 5
        assert (
            "CMIP6_ScenarioMIP_NCC_NorESM2-MM_ssp126_r1i1p1f1_example-region"
            in out.keys()
        )

    @pytest.mark.parametrize("allow_resampling", [True, False])
    def test_allow_resampling(self, allow_resampling):
        out = xs.search_data_catalogs(
            data_catalogs=deepcopy(self.small_cat),
            variables_and_freqs={"tas": "YS"},
            allow_resampling=allow_resampling,
        )
        assert len(out) == (5 if allow_resampling else 0)

    @pytest.mark.parametrize(
        "restrict_warming_level",
        [
            True,
            {"ignore_member": True},
            {"wl": 2},
            {"wl": 3},
            {"wl": 2, "ignore_member": True},
        ],
    )
    def test_warminglevel(self, restrict_warming_level):
        out = xs.search_data_catalogs(
            data_catalogs=self.small_cat,
            variables_and_freqs={"tas": "D"},
            restrict_warming_level=restrict_warming_level,
        )
        assert (
            out == 5
            if restrict_warming_level == {"ignore_member": True}
            else 4
            if (
                (restrict_warming_level is True)
                or (restrict_warming_level == {"wl": 2, "ignore_member": True})
            )
            else 3
            if restrict_warming_level == {"wl": 2}
            else 2
        )

    @pytest.mark.parametrize("restrict_resolution", [None, "finest"])
    def test_restrict_resolution(self, restrict_resolution):
        out = xs.search_data_catalogs(
            data_catalogs=self.big_cat,
            variables_and_freqs={"tas": "D"},
            other_search_criteria={
                "institution": ["NOAA-GFDL"],
                "experiment": ["ssp585"],
            },
            restrict_resolution=restrict_resolution,
        )
        assert len(out) == 3 if restrict_resolution is None else 2

    @pytest.mark.parametrize("restrict_members", [None, {"ordered": 5}])
    def test_restrict_members(self, restrict_members):
        out = xs.search_data_catalogs(
            data_catalogs=self.big_cat,
            variables_and_freqs={"tas": "D"},
            other_search_criteria={"source": ["CanESM5"], "experiment": ["ssp585"]},
            restrict_members=restrict_members,
        )
        assert len(out) == (50 if restrict_members is None else 5)
        if restrict_members is not None:
            assert all(
                o in out.keys()
                for o in [
                    "ScenarioMIP_CCCma_CanESM5_ssp585_r1i1p2f1_gn",
                    "ScenarioMIP_CCCma_CanESM5_ssp585_r1i1p1f1_gn",
                    "ScenarioMIP_CCCma_CanESM5_ssp585_r2i1p1f1_gn",
                    "ScenarioMIP_CCCma_CanESM5_ssp585_r2i1p2f1_gn",
                    "ScenarioMIP_CCCma_CanESM5_ssp585_r3i1p1f1_gn",
                ]
            )

        assert (
            len(
                xs.search_data_catalogs(
                    data_catalogs=self.big_cat,
                    variables_and_freqs={"tas": "D"},
                    other_search_criteria={
                        "institution": ["NOAA-GFDL"],
                        "experiment": ["ssp585"],
                    },
                    restrict_members=restrict_members,
                )
            )
            == 3
        )

    @pytest.mark.parametrize("allow_conversion", [True, False])
    def test_allow_conversion(self, allow_conversion):
        out = xs.search_data_catalogs(
            data_catalogs=self.big_cat,
            variables_and_freqs={"evspsblpot": "D"},
            other_search_criteria={
                "institution": ["NOAA-GFDL"],
                "experiment": ["ssp585"],
            },
            allow_conversion=allow_conversion,
        )
        assert len(out) == (3 if allow_conversion else 0)
        if allow_conversion:
            assert all(
                v in out[list(out.keys())[0]].unique("variable")
                for v in ["tasmin", "tasmax"]
            )
            assert "evspsblpot" not in out[list(out.keys())[0]].unique("variable")

    def test_no_match(self):
        out = xs.search_data_catalogs(
            data_catalogs=self.small_cat,
            variables_and_freqs={"tas": "YS"},
            allow_resampling=False,
        )
        assert isinstance(out, dict)
        assert len(out) == 0
        out = xs.search_data_catalogs(
            data_catalogs=self.small_cat,
            variables_and_freqs={"tas": "D"},
            other_search_criteria={"experiment": "not_real"},
        )
        assert isinstance(out, dict)
        assert len(out) == 0

    def test_input_types(self):
        # data_catalogs_1 = notebooks / "samples" / "tutorial-catalog.json"
        # data_catalogs_2 = notebooks / "samples" / "pangeo-cmip6.json"
        # out = xs.search_data_catalogs(
        #     data_catalogs=data_catalogs_1,
        #     variables_and_freqs={"tas": "D"},
        #     other_search_criteria={
        #         "experiment": "ssp585",
        #         "source": "NorESM.*",
        #         "member": "r1i1p1f1",
        #     },
        # )
        pass

    def test_match_histfut(self):
        out = xs.search_data_catalogs(
            data_catalogs=self.big_cat,
            variables_and_freqs={"tas": "D"},
            other_search_criteria={"experiment": "ssp585", "source": "CanESM5"},
            restrict_members={"ordered": 1},
            match_hist_and_fut=True,
        )
        assert (
            str(
                sorted(
                    out["ScenarioMIP_CCCma_CanESM5_ssp585_r1i1p1f1_gn"].unique(
                        "date_start"
                    )
                )[0]
            )
            == "1985-01-01 00:00:00"
        )
        assert (
            str(
                sorted(
                    out["ScenarioMIP_CCCma_CanESM5_ssp585_r1i1p1f1_gn"].unique(
                        "date_start"
                    )
                )[1]
            )
            == "2015-01-01 00:00:00"
        )

    def test_fx(self):
        pass


class TestGetWarmingLevel:
    def test_list(self):
        out = xs.get_warming_level(
            ["CMIP6_CanESM5_ssp126_r1i1p1f1", "CMIP6_CanESM5_ssp245_r1i1p1f1"],
            wl=2,
            window=20,
            return_horizon=False,
        )
        assert isinstance(out, dict)
        assert out["CMIP6_CanESM5_ssp126_r1i1p1f1"] == "2026"

    def test_string_with_horizon(self):
        out = xs.get_warming_level(
            "CMIP6_CanESM5_ssp585_r1i1p1f1", wl=2, window=20, return_horizon=True
        )
        assert isinstance(out, list)
        assert out == ["2013", "2032"]

    @pytest.mark.parametrize("attrs", ["global", "regional", "regional_w_institution"])
    def test_ds(self, attrs):
        ds = timeseries(
            np.tile(np.arange(1, 2), 30),
            variable="tas",
            start="2001-01-01",
            freq="AS-JAN",
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
        assert (
            xs.get_warming_level(
                ds, wl=2, window=20, ignore_member=True, return_horizon=False
            )
            == "2026"
        )

    def test_multiple_matches(self):
        # 55 instances of CanESM2-rcp85 in the CSV, but it should still return a single value
        assert (
            xs.get_warming_level(
                "CMIP5_CanESM2_rcp85_.*", wl=3.5, window=30, return_horizon=False
            )
            == "2059"
        )

    def test_odd_window(self):
        assert xs.get_warming_level(
            "CMIP6_CanESM5_ssp126_r1i1p1f1", wl=2, window=21, return_horizon=True
        ) == ["2016", "2036"]

    def test_none(self):
        assert (
            xs.get_warming_level(
                "CMIP6_CanESM5_ssp585_r1i1p1f1", wl=20, window=20, return_horizon=False
            )
            is None
        )
        assert (
            xs.get_warming_level(
                "CMIP6_notreal_ssp585_r1i1p1f1", wl=20, window=20, return_horizon=False
            )
            is None
        )

    def test_wrong_types(self):
        with pytest.raises(ValueError):
            xs.get_warming_level(
                {"this": "is not valid."}, wl=2, window=20, return_horizon=True
            )
        with pytest.raises(ValueError):
            xs.get_warming_level(
                "CMIP6_CanESM5_ssp585_r1i1p1f1_toomany_underscores",
                wl=2,
                window=20,
                return_horizon=True,
            )
        with pytest.raises(ValueError):
            xs.get_warming_level(
                "CMIP6_CanESM5_ssp585_r1i1p1f1", wl=2, window=3.85, return_horizon=True
            )


class TestSubsetWarmingLevel:
    ds = timeseries(
        np.tile(np.arange(1, 2), 50),
        variable="tas",
        start="2000-01-01",
        freq="AS-JAN",
        as_dataset=True,
    )
    ds.attrs = {
        "cat:mip_era": "CMIP6",
        "cat:source": "CanESM5",
        "cat:experiment": "ssp585",
        "cat:member": "r1i1p1f1",
    }

    def test_default(self):
        ds_sub = xs.subset_warming_level(TestSubsetWarmingLevel.ds, wl=2)

        np.testing.assert_array_equal(ds_sub.time.dt.year[0], 2013)
        np.testing.assert_array_equal(ds_sub.time.dt.year[-1], 2032)
        np.testing.assert_array_equal(ds_sub.warminglevel[0], "+2Cvs1850-1900")
        assert ds_sub.warminglevel.attrs["baseline"] == "1850-1900"
        assert ds_sub.attrs["cat:processing_level"] == "warminglevel-2vs1850-1900"

    def test_kwargs(self):
        ds_sub = xs.subset_warming_level(
            TestSubsetWarmingLevel.ds,
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

    def test_outofrange(self):
        assert xs.subset_warming_level(TestSubsetWarmingLevel.ds, wl=5) is None

    def test_none(self):
        assert xs.subset_warming_level(TestSubsetWarmingLevel.ds, wl=20) is None
