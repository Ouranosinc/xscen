import numpy as np
import pytest
import xarray as xr
import xclim as xc
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs


class TestHealthChecks:
    @pytest.mark.parametrize("good", [True, False])
    def test_structure(self, datablock_3d):
        ds = datablock_3d(
            np.random.rand(5, 5, 5), "tas", "rlon", 0, "rlat", 0, as_dataset=True
        )

        # Right structure
        xs.diagnostics.health_checks(
            ds,
            structure={
                "dims": ["time", "rlat", "rlon"],
                "coords": ["time", "rlat", "rlon", "lon", "lat"],
            },
            raise_on=["all"],
        )

        # Wrong structures
        with pytest.raises(ValueError, match="The dimension 'lat' is missing."):
            xs.diagnostics.health_checks(
                ds, structure={"dims": ["time", "lat", "rlon"]}, raise_on=["structure"]
            )
        with pytest.raises(ValueError, match="The coordinate 'another' is missing."):
            xs.diagnostics.health_checks(
                ds, structure={"coords": ["another"]}, raise_on=["structure"]
            )
        with pytest.raises(
            UserWarning, match="'tas' is detected as a data variable, not a coordinate."
        ):
            xs.diagnostics.health_checks(
                ds, structure={"coords": ["tas"]}, raise_on=["structure"]
            )
        with pytest.warns(
            UserWarning, match="'tas' is detected as a coordinate, not a data variable."
        ):
            xs.diagnostics.health_checks(ds, structure={"coords": ["tas"]})

    @pytest.mark.parametrize("cal", ["default", "standard", "365_day"])
    def test_calendar(self, cal):
        ds = timeseries(np.arange(0, 365), "tas", "1/1/2000", freq="D", as_dataset=True)
        if cal != "default":
            ds = ds.convert_calendar(cal)

        calendars = ["standard", "proleptic_gregorian", "noleap", "365_day", "360_day"]
        fail_on = [0, 1, 4] if cal == "365_day" else [2, 3, 4]
        for i, cals in enumerate(calendars):
            if i in fail_on:
                with pytest.raises(ValueError, match=f"The calendar is not '{cals}'."):
                    xs.diagnostics.health_checks(ds, calendar=cals, raise_on=["all"])
            else:
                xs.diagnostics.health_checks(ds, calendar=cals, raise_on=["all"])

    @pytest.mark.parametrize("cal", ["default", "360_day", "365_day"])
    def test_dates(self, cal):
        ds = timeseries(np.arange(0, 365), "tas", "1/1/2000", freq="D", as_dataset=True)
        if cal != "default":
            ds = ds.convert_calendar(cal, align_on="date")

        # Right dates
        xs.diagnostics.health_checks(
            ds, start_date="2000-01-01", end_date="2000-12-30", raise_on=["all"]
        )
        xs.diagnostics.health_checks(ds, start_date="2000-01-02", raise_on=["all"])
        xs.diagnostics.health_checks(ds, end_date="2000-01-02", raise_on=["all"])
        if cal == "360_day":
            xs.diagnostics.health_checks(ds, end_date="2000-02-30", raise_on=["all"])

        # Wrong dates
        with pytest.raises(
            ValueError, match="The start date is not at least 1999-01-02."
        ):
            xs.diagnostics.health_checks(ds, start_date="1999-01-02", raise_on=["all"])
        with pytest.raises(
            ValueError, match="The end date is not at least 2001-01-01."
        ):
            xs.diagnostics.health_checks(
                ds, end_date="2001-01-01", raise_on=["end_date"]
            )

    def test_variables(self):
        ds = timeseries(np.arange(0, 365), "tas", "1/1/2000", freq="D")

        # Right units
        xs.diagnostics.health_checks(
            ds, variables_and_units={"tas": "K"}, raise_on=["all"]
        )

        # Wrong variables or units
        with pytest.raises(ValueError, match="The variable 'tas2' is missing."):
            xs.diagnostics.health_checks(
                ds, variables_and_units={"tas2": "K"}, raise_on=["all"]
            )
        with pytest.warns(
            UserWarning,
            match="The variable 'tas' does not have the expected units 'degC'. Received 'K'.",
        ):
            xs.diagnostics.health_checks(ds, variables_and_units={"tas": "degC"})
        with pytest.raises(
            xc.core.utils.ValidationError,
            match="Data units kelvin are not compatible with requested 1 millimeter.",
        ):
            xs.diagnostics.health_checks(
                ds, variables_and_units={"tas": "mm"}
            )  # Should always raise on ValidationError

    def test_cfchecks(self):
        ds = timeseries(np.arange(0, 365), "tas", "1/1/2000", freq="D", as_dataset=True)

        # Good checks
        cfcheck = {
            "tas": {
                "check_valid": {"key": "standard_name", "expected": "air_temperature"},
                "cfcheck_from_name": {},
            }
        }
        xs.diagnostics.health_checks(ds, cfchecks=cfcheck, raise_on=["all"])

        # Bad checks
        cfcheck = {
            "tas": {
                "check_valid": {"key": "standard_name", "expected": ["not", "good"]},
                "cfcheck_from_name": {"varname": "pr"},
            }
        }
        with pytest.raises(
            ValueError, match="['not', 'good']"
        ):  # Will raise on the first check
            xs.diagnostics.health_checks(ds, cfchecks=cfcheck, raise_on=["all"])
        with pytest.warns(
            UserWarning, match="['precipitation_flux']"
        ):  # Make sure the second check is still run
            xs.diagnostics.health_checks(ds, cfchecks=cfcheck)

    @pytest.mark.parametrize(
        "freq, gap", [("D", False), ("MS", False), ("3H", False), ("D", True)]
    )
    def test_freq(self, freq, gap):
        ds = timeseries(
            np.arange(0, 365), "tas", "1/1/2000", freq=freq, as_dataset=True
        )
        if gap is False:
            # Right frequency
            xs.diagnostics.health_checks(ds, freq=freq, raise_on=["all"])

            # Wrong frequency
            with pytest.raises(ValueError, match="The frequency is not 'M'."):
                xs.diagnostics.health_checks(ds, freq="M", raise_on=["all"])
        else:
            ds = xr.concat(
                [ds.isel(time=slice(0, 100)), ds.isel(time=slice(200, 365))], dim="time"
            )
            with pytest.raises(
                ValueError,
                match="The timesteps are irregular or cannot be inferred by xarray.",
            ):
                xs.diagnostics.health_checks(ds, freq="D", raise_on=["all"])
            with pytest.warns(
                UserWarning,
                match="Frequency None is not supported for missing data checks. That check will be skipped.",
            ):
                xs.diagnostics.health_checks(ds, missing="any")

    @pytest.mark.parametrize("missing", ["missing_any", "missing_wmo"])
    def test_missing(self, missing):
        ds = timeseries(
            np.tile(np.arange(1, 366), 3), "tas", "1/1/2001", freq="D", as_dataset=True
        )
        ds = ds.where((ds.time.dt.year > 2001) | (ds.time.dt.dayofyear > 2))

        if missing == "missing_any":
            with pytest.raises(
                ValueError, match="The variable 'tas' has missing values."
            ):
                xs.diagnostics.health_checks(ds, missing=missing, raise_on=["all"])
        else:
            xs.diagnostics.health_checks(ds, missing=missing, raise_on=["all"])

    @pytest.mark.parametrize("flag", ["good", "bad"])
    def test_flags(self, flag):
        tasmin = np.array([-15] * 365 * 3)
        tasmax = np.array([15] * 365 * 3)

        if flag == "good":
            flags = {"tasmax": {"tasmax_below_tasmin": {}}}
            ds = timeseries(tasmin, "tasmin", "1/1/2000", freq="D", as_dataset=True)
            ds["tasmax"] = timeseries(tasmax, "tasmax", "1/1/2000", freq="D")
            xs.diagnostics.health_checks(ds, flags=flags)
            # xs.diagnostics.health_checks(ds, flags=flags, raise_on=["all"])  # FIXME: This always raises an error
        else:
            tasmin[7] = 150
            flags = {
                "tasmin": {
                    "temperature_extremely_high": {},
                    "values_repeating_for_n_or_more_days": {"n": 10},
                },
                "tasmax": {
                    "tasmax_below_tasmin": {},
                    "values_repeating_for_n_or_more_days": {"n": 10},
                },
            }
            ds = timeseries(tasmin, "tasmin", "1/1/2000", freq="D", as_dataset=True)
            ds["tasmax"] = timeseries(tasmax, "tasmax", "1/1/2000", freq="D")
            with pytest.raises(
                xc.core.dataflags.DataQualityException,
                match="Runs of repetitive values for 10 or more days found for tasmin.",
            ):
                xs.diagnostics.health_checks(ds, flags=flags, raise_on=["all"])
            with pytest.warns(UserWarning, match="tasmax_below_tasmin"):
                xs.diagnostics.health_checks(ds, flags=flags)
                dsflags = xs.diagnostics.health_checks(
                    ds, flags=flags, flags_kwargs={"freq": "D"}, return_flags=True
                )

                assert len(dsflags.time) == len(ds.time)
                assert len(dsflags.data_vars) == 4
                assert all(
                    [
                        v in dsflags.data_vars
                        for v in [
                            "tasmin_temperature_extremely_high",
                            "tasmin_values_repeating_for_10_or_more_days",
                            "tasmax_tasmax_below_tasmin",
                            "tasmax_values_repeating_for_10_or_more_days",
                        ]
                    ]
                )