from copy import deepcopy

import numpy as np
import pytest
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs


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
