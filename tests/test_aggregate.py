import cftime
import numpy as np
import pytest
import xarray as xr
import xclim
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs


class TestClimatologicalMean:
    def test_daily(self):
        ds = timeseries(
            np.tile(np.arange(1, 13), 3),
            variable="tas",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
        )
        with pytest.raises(NotImplementedError):
            xs.climatological_mean(ds)

    def test_all_default(self):
        for xrfreq in ["MS", "AS-JAN"]:
            o = 12 if xrfreq == "MS" else 1

            ds = timeseries(
                np.tile(np.arange(1, o + 1), 30),
                variable="tas",
                start="2001-01-01",
                freq=xrfreq,
                as_dataset=True,
            )
            out = xs.climatological_mean(ds)

            np.testing.assert_array_equal(out.tas, np.arange(1, o + 1))
            np.testing.assert_array_equal(
                len(out.time), o * len(np.unique(out.horizon.values))
            )
            np.testing.assert_array_equal(out.time[0], ds.time[0])
            assert (
                out.tas.attrs["description"]
                == f"30-year mean of {ds.tas.attrs['description']}"
            )
            assert (
                "30-year rolling average (non-centered) with a minimum of 30 years of data"
                in out.tas.attrs["history"]
            )
            assert all(out.horizon.values == "2001-2030")
            assert out.attrs["cat:processing_level"] == "climatology"

    def test_options(self):
        for xrfreq in ["MS", "AS-JAN"]:
            o = 12 if xrfreq == "MS" else 1

            ds = timeseries(
                np.tile(np.arange(1, o + 1), 30),
                variable="tas",
                start="2001-01-01",
                freq=xrfreq,
                as_dataset=True,
            )
            out = xs.climatological_mean(
                ds, window=15, interval=5, to_level="for_testing"
            )

            np.testing.assert_array_equal(
                out.tas,
                np.tile(np.arange(1, o + 1), len(np.unique(out.horizon.values))),
            )
            np.testing.assert_array_equal(
                len(out.time), o * len(np.unique(out.horizon.values))
            )
            np.testing.assert_array_equal(out.time[0], ds.time[0])
            assert (
                out.tas.attrs["description"]
                == f"15-year mean of {ds.tas.attrs['description']}"
            )
            assert (
                "15-year rolling average (non-centered) with a minimum of 15 years of data"
                in out.tas.attrs["history"]
            )
            assert all(
                h in np.unique(out.horizon.values)
                for h in ["2001-2015", "2006-2020", "2011-2025", "2016-2030"]
            )
            assert out.attrs["cat:processing_level"] == "for_testing"

    def test_minperiods(self):
        ds = timeseries(
            np.tile(np.arange(1, 5), 30),
            variable="tas",
            start="2000-12-01",
            freq="QS-DEC",
            as_dataset=True,
        )
        ds = ds.where(ds.time.dt.year >= 2001)
        out = xs.climatological_mean(ds, window=30)
        assert all(np.isreal(out.tas))
        assert len(out.time) == 8
        out = xs.climatological_mean(ds, window=30, min_periods=30)
        assert np.sum(np.isreal(out.tas))

        with pytest.raises(ValueError):
            xs.climatological_mean(ds, window=5, min_periods=6)

    def test_periods(self):
        ds1 = timeseries(
            np.tile(np.arange(1, 2), 10),
            variable="tas",
            start="2001-01-01",
            freq="AS-JAN",
            as_dataset=True,
        )
        ds2 = timeseries(
            np.tile(np.arange(1, 2), 10),
            variable="tas",
            start="2021-01-01",
            freq="AS-JAN",
            as_dataset=True,
        )
        ds = xr.concat([ds1, ds2], dim="time")
        with pytest.raises(ValueError):
            xs.climatological_mean(ds)

        out = xs.climatological_mean(ds, periods=[["2001", "2010"], ["2021", "2030"]])
        np.testing.assert_array_equal(len(out.time), 2)
        assert all(
            h in np.unique(out.horizon.values) for h in ["2001-2010", "2021-2030"]
        )

    def test_calendars(self):
        ds = timeseries(
            np.tile(np.arange(1, 2), 30),
            variable="tas",
            start="2001-01-01",
            freq="AS-JAN",
            as_dataset=True,
        )
        time_coord = [
            xclim.core.calendar.datetime_classes["noleap"](y, m, d)
            for y, m, d in zip(
                ds.time.dt.year.values,
                ds.time.dt.month.values,
                ds.time.dt.day.values,
            )
        ]
        ds["time"] = time_coord
        out = xs.climatological_mean(ds)

        assert isinstance(out.time.values[0], cftime.DatetimeNoLeap)

    # def test_monthly_all_default(self, tas_series):
    #     ds = tas_series(
    #         np.tile(np.arange(1, 13), 3), start="2001-01-01", freq="MS", as_dataset=True
    #     )
    #     out = xs.climatological_mean(ds)
    #     assert out.attrs["cat:processing_level"] == "climatology"
    #     np.testing.assert_array_equal(out.tas, np.arange(1, 13))

    # test various options
    # test not continuous
    # test calendars
    # test multiple periods
