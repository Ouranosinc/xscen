import numpy as np
import pytest
import xarray as xr
import xclim as xc
from conftest import notebooks
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs
from xscen.testing import datablock_3d


class TestHealthChecks:
    def test_structure(self, datablock_3d):
        ds = datablock_3d(np.random.rand(5, 5, 5), "tas", "rlon", 0, "rlat", 0, as_dataset=True)

        # Right structure
        xs.diagnostics.health_checks(
            ds,
            structure={
                "dims": ["time", "rlat", "rlon"],
                "coords": ["time", "rlat", "rlon", "lon", "lat", "rotated_pole"],
            },
            raise_on=["all"],
        )

        # Wrong structures
        with pytest.raises(ValueError, match="The dimension 'lat' is missing."):
            xs.diagnostics.health_checks(ds, structure={"dims": ["time", "lat", "rlon"]}, raise_on=["all"])
        with pytest.warns(UserWarning, match="The dimension 'lat' is missing."):
            xs.diagnostics.health_checks(ds, structure={"dims": ["time", "lat", "rlon"]})
        with pytest.raises(ValueError, match="The coordinate 'another' is missing."):
            xs.diagnostics.health_checks(ds, structure={"coords": ["another"]}, raise_on=["structure"])
        with pytest.warns(UserWarning, match="The coordinate 'another' is missing."):
            xs.diagnostics.health_checks(ds, structure={"coords": ["another"]})
        with pytest.raises(ValueError, match="'tas' is detected as a data variable, not a coordinate."):
            xs.diagnostics.health_checks(ds, structure={"coords": ["tas"]}, raise_on=["structure"])
        with pytest.warns(UserWarning, match="'tas' is detected as a data variable, not a coordinate."):
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
                with pytest.warns(UserWarning, match=f"The calendar is not '{cals}'."):
                    xs.diagnostics.health_checks(ds, calendar=cals)
            else:
                xs.diagnostics.health_checks(ds, calendar=cals, raise_on=["all"])

    @pytest.mark.parametrize("cal", ["default", "360_day", "365_day"])
    def test_dates(self, cal):
        ds = timeseries(np.arange(0, 365), "tas", "1/1/2000", freq="D", as_dataset=True)
        if cal != "default":
            ds = ds.convert_calendar(cal, align_on="date")

        # Right dates
        xs.diagnostics.health_checks(ds, start_date="2000-01-01", end_date="2000-12-30", raise_on=["all"])
        xs.diagnostics.health_checks(ds, start_date="2000-01-02", raise_on=["all"])
        xs.diagnostics.health_checks(ds, end_date="2000-01-02", raise_on=["all"])

        # Wrong dates
        with pytest.raises(ValueError, match="The start date is not at least 1999-01-02."):
            xs.diagnostics.health_checks(ds, start_date="1999-01-02", raise_on=["all"])
        with pytest.warns(UserWarning, match="The start date is not at least 1999-01-02."):
            xs.diagnostics.health_checks(ds, start_date="1999-01-02")
        with pytest.raises(ValueError, match="The end date is not at least 2001-01-01."):
            xs.diagnostics.health_checks(ds, end_date="2001-01-01", raise_on=["end_date"])
        with pytest.warns(UserWarning, match="The end date is not at least 2001-01-01."):
            xs.diagnostics.health_checks(ds, end_date="2001-01-01")

    def test_variables(self):
        ds = timeseries(np.arange(0, 365), "tas", "1/1/2000", freq="D")

        # Right units
        xs.diagnostics.health_checks(ds, variables_and_units={"tas": "K"}, raise_on=["all"])

        # Wrong variables or units
        with pytest.raises(ValueError, match="The variable 'tas2' is missing."):
            xs.diagnostics.health_checks(ds, variables_and_units={"tas2": "K"}, raise_on=["all"])
        with pytest.raises(
            ValueError,
            match="The variable 'tas' does not have the expected units 'degC'. Received 'K'.",
        ):
            xs.diagnostics.health_checks(ds, variables_and_units={"tas": "degC"}, raise_on=["all"])
        with pytest.warns(
            UserWarning,
            match="The variable 'tas' does not have the expected units 'degC'. Received 'K'.",
        ):
            xs.diagnostics.health_checks(ds, variables_and_units={"tas": "degC"})
        with pytest.warns(
            UserWarning,
            match="are not compatible with requested mm.",
        ):
            xs.diagnostics.health_checks(ds, variables_and_units={"tas": "mm"})

        # this gives "°C" as units
        ds1 = xc.units.convert_units_to(ds, "degC")
        # It should be okay by default
        xs.diagnostics.health_checks(ds1, variables_and_units={"tas": "degC"}, raise_on=["all"])
        xs.diagnostics.health_checks(ds1, variables_and_units={"tas": "°C"}, raise_on=["all"])
        # but raise an error if we want something stritcly the same
        with pytest.raises(
            ValueError,
            match="The variable 'tas' does not have the expected units 'degC'. Received '°C'.",
        ):
            xs.diagnostics.health_checks(
                ds1,
                variables_and_units={"tas": "degC"},
                strict_units=True,
                raise_on=["all"],
            )

    def test_cfchecks(self):
        ds = timeseries(np.arange(0, 365), "tas", "1/1/2000", freq="D", as_dataset=True)

        # Check that the cfchecks are valid
        with pytest.raises(ValueError, match="Check 'fake' is not in xclim."):
            xs.diagnostics.health_checks(ds, cfchecks={"tas": {"fake": {}}})

        # Good checks
        cfcheck = {
            "tas": {
                "check_valid": {"key": "standard_name", "expected": "air_temperature"},
                "cfcheck_from_name": {},
            }
        }
        xs.diagnostics.health_checks(ds, cfchecks=cfcheck, raise_on=["all"])

        # Bad checks
        bad_cfcheck = {
            "tas": {
                "check_valid": {"key": "standard_name", "expected": "something_else"},
                "cfcheck_from_name": {"varname": "pr"},
            }
        }
        with pytest.raises(ValueError, match="['something_else']"):  # Will raise on the first check
            xs.diagnostics.health_checks(ds, cfchecks=bad_cfcheck, raise_on=["all"])
        with pytest.warns(UserWarning, match="['precipitation_flux']"):  # Make sure the second check is still run
            xs.diagnostics.health_checks(ds, cfchecks=bad_cfcheck)

    @pytest.mark.parametrize("freq, gap", [("D", False), ("MS", False), ("3h", False), ("D", True)])
    def test_freq(self, freq, gap):
        ds = timeseries(np.arange(0, 365), "tas", "1/1/2000", freq=freq, as_dataset=True)
        if gap is False:
            # Right frequency
            xs.diagnostics.health_checks(ds, freq=freq, raise_on=["all"])

            # Wrong frequency
            with pytest.raises(ValueError, match="The frequency is not 'M'."):
                xs.diagnostics.health_checks(ds, freq="M", raise_on=["all"])
            with pytest.warns(UserWarning, match="The frequency is not 'M'."):
                xs.diagnostics.health_checks(ds, freq="M")
        else:
            ds = xr.concat([ds.isel(time=slice(0, 100)), ds.isel(time=slice(200, 365))], dim="time")
            with pytest.raises(
                ValueError,
                match="The timesteps are irregular or cannot be inferred by xarray.",
            ):
                xs.diagnostics.health_checks(ds, freq="D", raise_on=["all"])
            with pytest.warns(
                UserWarning,
                match="The timesteps are irregular or cannot be inferred by xarray.",
            ):
                xs.diagnostics.health_checks(ds, freq="D")
            with pytest.warns(
                UserWarning,
                match="Frequency None is not supported for missing data checks. That check will be skipped.",
            ):
                xs.diagnostics.health_checks(ds, missing="any")

    @pytest.mark.parametrize("missing", ["missing_any", "missing_wmo", "both"])
    def test_missing(self, missing):
        ds = timeseries(np.tile(np.arange(1, 366), 3), "tas", "1/1/2001", freq="D", as_dataset=True)
        ds = ds.where((ds.time.dt.year > 2001) | (ds.time.dt.dayofyear > 2))

        if missing == "both":
            missing = ["missing_any", "missing_wmo"]

        if missing == "missing_any" or isinstance(missing, list):
            with pytest.raises(
                ValueError,
                match="The variable 'tas' has missing values according to the 'missing_any' method.",
            ):
                xs.diagnostics.health_checks(ds, missing=missing, raise_on=["all"])
            with pytest.warns(
                UserWarning,
                match="The variable 'tas' has missing values according to the 'missing_any' method.",
            ):
                xs.diagnostics.health_checks(ds, missing=missing)
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
            xs.diagnostics.health_checks(ds, flags=flags, raise_on=["all"])
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
                ValueError,
                match="tasmax_below_tasmin",
            ):
                xs.diagnostics.health_checks(ds, flags=flags, raise_on=["all"])
            with pytest.warns(UserWarning, match="tasmax_below_tasmin"):
                xs.diagnostics.health_checks(ds, flags=flags)
                dsflags = xs.diagnostics.health_checks(ds, flags=flags, flags_kwargs={"freq": "D"}, return_flags=True)

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


class TestPropertiesMeasures:
    yaml_file = notebooks / "samples" / "properties.yml"
    ds = timeseries(np.ones(365 * 3), variable="tas", start="2001-01-01", freq="D", as_dataset=True)

    @pytest.mark.parametrize("input", ["module", "iter"])
    def test_input_types(self, input):
        module = xs.indicators.load_xclim_module(self.yaml_file)
        p1, m1 = xs.properties_and_measures(
            self.ds,
            properties=module if input == "module" else module.iter_indicators(),
        )
        p2, m2 = xs.properties_and_measures(self.ds, properties=self.yaml_file)
        assert p1.equals(p2)
        assert m1.equals(m2)

    @pytest.mark.parametrize("to_level", [None, "test"])
    def test_level(self, to_level):
        if to_level is None:
            p, m = xs.properties_and_measures(self.ds, properties=self.yaml_file)
            assert "diag-properties" == p.attrs["cat:processing_level"]
            assert "diag-measures" == m.attrs["cat:processing_level"]

        else:
            p, m = xs.properties_and_measures(
                self.ds,
                properties=self.yaml_file,
                to_level_prop=to_level,
                to_level_meas=to_level,
            )
            assert to_level == p.attrs["cat:processing_level"]
            assert to_level == m.attrs["cat:processing_level"]

    @pytest.mark.parametrize("period", [None, ["2001", "2001"]])
    def test_output(self, period):
        values = np.ones(365 * 2)
        values[:365] = 2
        ds = timeseries(values, variable="tas", start="2001-01-01", freq="D", as_dataset=True)
        ds["da"] = ds.tas

        p, m = xs.properties_and_measures(
            ds,
            properties=self.yaml_file,
            period=period,
        )

        if period is None:
            np.testing.assert_allclose(p["quantile_98_tas"].values, 2)
            np.testing.assert_allclose(p["mean-tas"].values, 1.5)
        else:
            np.testing.assert_allclose(p["quantile_98_tas"].values, 2)
            np.testing.assert_allclose(p["mean-tas"].values, 2)

    def test_unstack(self):
        ds = datablock_3d(
            np.array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]]] * 3, "float"),
            "tas",
            "lon",
            -70,
            "lat",
            15,
            30,
            30,
            as_dataset=True,
        )

        ds_stack = xs.utils.stack_drop_nans(
            ds,
            mask=xr.where(ds.tas.isel(time=0).isnull(), False, True).drop_vars("time"),
        )

        p, m = xs.properties_and_measures(
            ds_stack,
            properties=self.yaml_file,
            unstack=True,
        )

        assert "lat" in p.dims
        assert "lon" in p.dims
        assert "loc" not in p.dims

    def test_rechunk(self):
        ds = datablock_3d(
            np.array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]]] * 3, "float"),
            "tas",
            "lon",
            -70,
            "lat",
            15,
            30,
            30,
            as_dataset=True,
        )
        p, m = xs.properties_and_measures(
            ds,
            properties=self.yaml_file,
            rechunk={"lat": 1, "lon": 1},
        )

        assert p.chunks["lat"] == (1, 1, 1)
        assert p.chunks["lon"] == (1, 1, 1)

    def test_units(self):
        p, m = xs.properties_and_measures(
            self.ds,
            properties=self.yaml_file,
            change_units_arg={"tas": "°C"},
        )

        assert p["mean-tas"].attrs["units"] == "°C"

    def test_dref_for_measure(self):
        p1, m1 = xs.properties_and_measures(
            self.ds,
            properties=self.yaml_file,
        )

        p2, m2 = xs.properties_and_measures(
            self.ds,
            properties=self.yaml_file,
            dref_for_measure=p1,
        )
        print(m2)
        print(m2["maximum_length_of_warm_spell"].values)
        assert m1.sizes == {}
        np.testing.assert_allclose(m2["maximum_length_of_warm_spell"].values, 0)

    def test_measures_heatmap(self):
        p1, m1 = xs.properties_and_measures(
            self.ds,
            properties=self.yaml_file,
        )

        p2, m2 = xs.properties_and_measures(
            self.ds,
            properties=self.yaml_file,
            dref_for_measure=p1,
        )

        out = xs.diagnostics.measures_heatmap({"m2": m2}, to_level="test")

        assert out.attrs["cat:processing_level"] == "test"
        assert "m2" in out.realization.values
        assert "mean-tas" in out.properties.values
        np.testing.assert_allclose(out["heatmap"].values, 0.5)

    def test_measures_improvement(self):
        p1, m1 = xs.properties_and_measures(
            self.ds,
            properties=self.yaml_file,
            period=["2001", "2001"],
        )

        p2, m2 = xs.properties_and_measures(
            self.ds,
            properties=self.yaml_file,
            dref_for_measure=p1,
            period=["2001", "2001"],
        )
        with pytest.warns(
            UserWarning,
            match="meas_datasets has more than 2 datasets." + " Only the first 2 will be compared.",
        ):
            out = xs.diagnostics.measures_improvement([m2, m2, m2], to_level="test")

        assert out.attrs["cat:processing_level"] == "test"
        assert "mean-tas" in out.properties.values
        np.testing.assert_allclose(out["improved_grid_points"].values, 1)

    def test_measures_improvement_dim(self):
        p1, m1 = xs.properties_and_measures(
            self.ds,
            properties=self.yaml_file,
            period=["2001", "2001"],
        )

        p2, m2 = xs.properties_and_measures(
            self.ds,
            properties=self.yaml_file,
            dref_for_measure=p1,
            period=["2001", "2001"],
        )
        m3 = xr.concat(
            [m2.quantile_98_tas.to_dataset().expand_dims({"dummy_dim": [i]}) for i in range(3)],
            dim="dummy_dim",
        )
        out = xs.diagnostics.measures_improvement([m3, m3], dim="dummy_dim", to_level="test")
        assert set(out.dims) == {"properties", "season"}
        np.testing.assert_allclose(out["improved_grid_points"].values, 1)

    def test_measures_improvement_2d(self):
        p1, m1 = xs.properties_and_measures(
            self.ds,
            properties=self.yaml_file,
        )

        p2, m2 = xs.properties_and_measures(
            self.ds,
            properties=self.yaml_file,
            dref_for_measure=p1,
        )

        imp = xs.diagnostics.measures_improvement([m2, m2], to_level="test")

        out = xs.diagnostics.measures_improvement_2d({"i1": imp, "i2": imp}, to_level="test")

        assert out.attrs["cat:processing_level"] == "test"
        assert "mean-tas" in out.properties.values
        assert "i1" in out.realization.values
        assert "i2" in out.realization.values
        np.testing.assert_allclose(out["improved_grid_points"].values, 1)

        out2 = xs.diagnostics.measures_improvement_2d(
            {
                "i1": [m2, m2],
                "i2": {"a": m2, "b": m2},
            },
            to_level="test",
        )

        assert out.equals(out2)
