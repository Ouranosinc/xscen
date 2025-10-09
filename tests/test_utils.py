from datetime import datetime
from pathlib import Path

import cftime
import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xclim as xc
import xsdba
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs
from xscen.testing import datablock_3d


class TestLocale:
    def test_update(self):
        ds = timeseries(
            np.tile(np.arange(1, 366), 30),
            variable="tas",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
        )
        indicator = xc.core.indicator.Indicator.from_dict(
            data={"base": "tg_mean"},
            identifier="tg_mean",
            module="atmos",
        )
        with xc.set_options(metadata_locales="fr"):
            out = xs.compute_indicators(ds, [("tg_mean", indicator)])["YS-JAN"]
        out = xs.climatological_op(out, op="mean")

        assert out["tg_mean_clim_mean"].attrs["long_name"] == "30-year climatological average of Mean daily mean temperature."
        assert out["tg_mean_clim_mean"].attrs["long_name_fr"] == "Moyenne 30 ans de Moyenne de la température moyenne quotidienne."

    @pytest.mark.parametrize("locale", ["fr", "jp"])
    def test_add(self, locale):
        # Dummy function to make gettext aware of translatable-strings
        def _(s):
            return s

        ds = timeseries(
            np.arange(1, 366),
            variable="tas",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
        )
        with xc.set_options(metadata_locales=locale):
            xs.utils.add_attr(ds, "some_attr", _("Ranking of measure performance"))
        assert ds.attrs["some_attr"] == "Ranking of measure performance"
        if locale == "fr":
            assert ds.attrs["some_attr_fr"] == "Classement de performance de la mesure"
        elif locale == "jp":
            # Japanese translation is not available, so the original string is used
            assert ds.attrs["some_attr_jp"] == "Ranking of measure performance"


class TestDateParser:
    @pytest.mark.parametrize(
        "date,end_of_period,dtype,exp",
        [
            ("2001", True, "datetime", pd.Timestamp("2001-12-31 23:59:59")),
            ("150004", True, "datetime", pd.Timestamp("1500-04-30 23:59:59")),
            ("31231212", None, "datetime", pd.Timestamp("3123-12-12")),
            ("2001-07-08", None, "period", pd.Period("2001-07-08", "h")),
            (pd.Timestamp("1993-05-20T12:07"), None, "str", "1993-05-20"),
            (
                cftime.Datetime360Day(1981, 2, 30),
                None,
                "datetime",
                pd.Timestamp("1981-02-28"),
            ),
            (
                np.datetime64("1200-11-12"),
                "Y",
                "datetime",
                pd.Timestamp("1200-12-31 23:59:59"),
            ),
            (
                datetime(2045, 5, 2, 13, 45),
                None,
                "datetime",
                pd.Timestamp("2045-05-02T13:45"),
            ),
            ("abc", None, "datetime", pd.Timestamp("NaT")),
            ("", True, "datetime", pd.Timestamp("NaT")),
            (
                pd.Period("2001-07-08", "h"),
                None,
                "datetime",
                pd.Timestamp("2001-07-08"),
            ),
        ],
    )
    def test_normal(self, date, end_of_period, dtype, exp):
        out = xs.utils.date_parser(date, end_of_period=end_of_period, out_dtype=dtype)
        if pd.isna(exp):
            assert pd.isna(out)
        else:
            assert out == exp


class TestMinCal:
    @pytest.mark.parametrize(
        "cals",
        [
            ["360_day", "365_day"],
            ["365_day", "default"],
            ["noleap", "default"],
            ["365_day", "noleap"],
            ["365_day", "all_leap"],
            ["366_day", "all_leap"],
            ["366_day", "default"],
        ],
    )
    def test_minimum_calendar(self, cals):
        out = xs.utils.minimum_calendar(cals)
        if "360_day" in cals:
            assert out == "360_day"
        elif any(c in cals for c in ["noleap", "365_day"]):
            assert out == "noleap"
        elif any(c in cals for c in ["default", "standard"]):
            assert out == "standard"
        else:
            assert out == "all_leap"

    def test_error(self):
        with pytest.warns(UserWarning, match="These calendars are not recognized"):
            xs.utils.minimum_calendar(["366_day", "foo"])


class TestTranslateTimeChunk:
    @pytest.mark.parametrize("chunk", [-1, 10])
    def test_normal(self, chunk):
        out = xs.utils.translate_time_chunk({"time": chunk, "lon": 50}, "noleap", 3450)
        assert out == {"time": 3450 if chunk == -1 else 10, "lon": 50}

    @pytest.mark.parametrize("calendar", ["360_day", "standard", "365_day", "366_day"])
    def test_ny(self, calendar):
        ndays = int(calendar.split("_")[0]) if "day" in calendar else 365.25
        out = xs.utils.translate_time_chunk({"time": "4year", "lon": 50}, calendar, 3450)
        assert out == {"time": ndays * 4, "lon": 50}

    def test_warning(self):
        with pytest.warns(UserWarning, match="The number of days"):
            xs.utils.translate_time_chunk({"time": "3year", "lon": 50}, "standard", 3450)

    def test_dict_of_dict(self):
        out = xs.utils.translate_time_chunk(
            {"tas": {"time": 10, "lon": 50}, "pr": {"time": -1, "lon": 50}},
            "noleap",
            3450,
        )
        assert out == {"tas": {"time": 10, "lon": 50}, "pr": {"time": 3450, "lon": 50}}


def test_naturalsort():
    assert xs.utils.natural_sort(["r1i1p1", "r2i1p1", "r10i1p1", "r1i1p2"]) == [
        "r1i1p1",
        "r1i1p2",
        "r2i1p1",
        "r10i1p1",
    ]


def get_cat_attrs():
    ds = timeseries(
        np.tile(np.arange(1, 2), 50),
        variable="tas",
        start="2000-01-01",
        freq="YS-JAN",
        as_dataset=True,
    )
    ds.attrs = {
        "foo": "bar",
        "cat:type": "simulation",
        "cat:variable": ("tas",),
        "dog:source": "CanESM5",
    }

    assert xs.utils.get_cat_attrs(ds) == {"type": "simulation", "variable": ("tas",)}
    assert xs.utils.get_cat_attrs(ds, var_as_str=True) == {
        "type": "simulation",
        "variable": "tas",
    }
    assert xs.utils.get_cat_attrs(ds, prefix="dog:") == {"source": "CanESM5"}
    assert xs.utils.get_cat_attrs(ds.attrs) == {
        "type": "simulation",
        "variable": ("tas",),
    }


class TestScripting:
    ds = timeseries(
        np.tile(np.arange(1, 2), 50),
        variable="tas",
        start="2000-01-01",
        freq="YS-JAN",
        as_dataset=True,
    )
    ds.attrs = {
        "cat:type": "simulation",
        "cat:processing_level": "raw",
        "cat:variable": ("tas",),
        "dog:source": "CanESM5",
    }

    @pytest.mark.parametrize(
        "ds, prefix, var_as_str",
        [["ds", "cat:", False], ["dict", "cat:", True], ["ds", "dog:", True]],
    )
    def test_get_cat_attrs(self, ds, prefix, var_as_str):
        data = self.ds if ds == "ds" else self.ds.attrs
        out = xs.utils.get_cat_attrs(data, prefix=prefix, var_as_str=var_as_str)

        if var_as_str and prefix == "cat:":
            assert out == {
                "type": "simulation",
                "processing_level": "raw",
                "variable": "tas",
            }
        elif not var_as_str and prefix == "cat:":
            assert out == {
                "type": "simulation",
                "processing_level": "raw",
                "variable": ("tas",),
            }
        elif prefix == "dog:":
            assert out == {"source": "CanESM5"}

    def test_strip_cat_attrs(self):
        out = xs.utils.strip_cat_attrs(self.ds)
        assert list(out.attrs.keys()) == ["dog:source"]


class TestStack:
    def test_no_nan(self):
        ds = datablock_3d(
            np.zeros((20, 10, 10)),
            "tas",
            "lon",
            -5,
            "lat",
            80.5,
            1,
            1,
            "2000-01-01",
            as_dataset=True,
        )
        mask = xr.where(ds.tas.isel(time=0).isnull(), False, True).drop_vars("time")
        out = xs.utils.stack_drop_nans(ds, mask=mask)
        assert "loc" in out.dims
        assert out.sizes["loc"] == 100

        ds_unstack = xs.utils.unstack_fill_nan(out)
        assert ds_unstack.equals(ds)

    def test_nan(self, tmp_path):
        data = np.zeros((20, 10, 10))
        data[:, 0, 0] = [np.nan] * 20
        ds = datablock_3d(
            data,
            "tas",
            "lon",
            -5,
            "lat",
            80.5,
            1,
            1,
            "2000-01-01",
            as_dataset=True,
        )

        mask = xr.where(ds.tas.isel(time=0).isnull(), False, True).drop_vars("time")
        ds.attrs["cat:domain"] = "RegionEssai"
        out = xs.utils.stack_drop_nans(
            ds,
            mask=mask,
            new_dim="loc1",
            to_file=str(tmp_path / "subfolder" / "coords_{domain}_{shape}.nc"),
        )
        assert "loc1" in out.dims
        assert out.sizes["loc1"] == 99
        assert (tmp_path / "subfolder" / "coords_RegionEssai_10x10.nc").is_file()

        out_no_mask = xs.utils.stack_drop_nans(ds, mask=["lon", "lat"], new_dim="loc1")
        assert out_no_mask.equals(out)

        ds_unstack = xs.utils.unstack_fill_nan(
            out,
            dim="loc1",
            coords=str(tmp_path / "subfolder" / "coords_{domain}_{shape}.nc"),
        )
        assert ds_unstack.equals(ds)

    @pytest.mark.parametrize("coords", ["file.nc", ["lon", "lat"], "dict", None])
    def test_fillnan_coords(self, tmpdir, coords):
        data = np.zeros((20, 10, 10))
        data[:, 1, 0] = [np.nan] * 20
        data[:, 0, :] = [np.nan] * 10
        ds = datablock_3d(
            data,
            "tas",
            "lon",
            -5,
            "lat",
            80.5,
            1,
            1,
            "2000-01-01",
            as_dataset=True,
        )
        ds.attrs["cat:domain"] = "RegionEssai"

        mask = xr.where(ds.tas.isel(time=0).isnull(), False, True).drop_vars("time")
        # Add mask as a coordinate
        ds = ds.assign_coords(z=mask.astype(int))
        ds.z.attrs["foo"] = "bar"

        if coords == "dict":
            coords = {"lon": ds.lon, "lat": ds.lat, "z": ds.z}
        elif coords == "file.nc":
            coords = str(tmpdir / "coords_{domain}_{shape}.nc")

        ds_stack = xs.utils.stack_drop_nans(ds, mask=mask, to_file=coords if isinstance(coords, str) else None)
        ds_unstack = xs.utils.unstack_fill_nan(
            ds_stack,
            coords=coords,
        )

        if isinstance(coords, list):
            # Cannot fully recover the original dataset.
            ds_unstack["z"] = ds_unstack["z"].fillna(0)
            assert ds_unstack.equals(ds.isel(lat=slice(1, None)))
        elif coords is None:
            # 'z' gets completely assigned as a dimension.
            assert "z" in ds_unstack.dims
            assert ds_unstack.isel(z=0).drop_vars("z").equals(ds.isel(lat=slice(1, None)).drop_vars("z"))
        else:
            assert ds_unstack.equals(ds)

    def test_maybe(self, tmp_path):
        data = np.zeros((20, 10, 10))
        data[:, 0, 0] = [np.nan] * 20
        ds = datablock_3d(
            data,
            "tas",
            "lon",
            -5,
            "lat",
            80.5,
            1,
            1,
            "2000-01-01",
            as_dataset=True,
        )
        mask = xr.where(ds.tas.isel(time=0).isnull(), False, True).drop_vars("time")
        ds.attrs["cat:domain"] = "RegionEssai"
        z = xr.DataArray(
            np.ones([10, 10]),
            dims=["lat", "lon"],
            coords={"lat": ds.lat, "lon": ds.lon},
        )
        z1d = xr.DataArray(np.ones([10]), dims=["lat"], coords={"lat": ds.lat})
        ds = ds.assign_coords(z=z, z1d=z1d)
        out = xs.utils.stack_drop_nans(
            ds,
            mask=mask,
            new_dim="loc1",
            to_file=str(tmp_path / "coords_{domain}_{shape}.nc"),
        )

        maybe_unstacked = xs.utils.maybe_unstack(out, dim="loc1", coords=str(tmp_path / "coords_{domain}_{shape}.nc"))
        assert maybe_unstacked.equals(out)
        # Call through clean_up to test the whole pipeline
        maybe_unstack_dict = {
            "dim": "loc1",
            "coords": str(tmp_path / "coords_{domain}_{shape}.nc"),
            "stack_drop_nans": True,
        }
        maybe_unstacked = xs.utils.clean_up(out, maybe_unstack_dict=maybe_unstack_dict)
        assert maybe_unstacked.equals(ds)
        maybe_unstacked = xs.utils.maybe_unstack(
            out,
            dim="loc1",
            coords=str(tmp_path / "coords_{domain}_{shape}.nc"),
            rechunk={"lon": -1, "lat": 2},
            stack_drop_nans=True,
        )
        assert dict(maybe_unstacked.chunks) == {
            "time": (20,),
            "lat": (2, 2, 2, 2, 2),
            "lon": (10,),
        }

    def test_maybe_default(self, tmp_path):
        data = np.zeros((20, 10, 10))
        data[:, 0, 0] = [np.nan] * 20
        ds = datablock_3d(
            data,
            "tas",
            "lon",
            -5,
            "lat",
            80.5,
            1,
            1,
            "2000-01-01",
            as_dataset=True,
        )
        mask = xr.where(ds.tas.isel(time=0).isnull(), False, True).drop_vars("time")
        ds.attrs["cat:domain"] = "RegionEssai"
        z = xr.DataArray(
            np.ones([10, 10]),
            dims=["lat", "lon"],
            coords={"lat": ds.lat, "lon": ds.lon},
        )
        z1d = xr.DataArray(np.ones([10]), dims=["lat"], coords={"lat": ds.lat})
        ds = ds.assign_coords(z=z, z1d=z1d)
        out = xs.utils.stack_drop_nans(
            ds,
            mask=mask,
            to_file=str(tmp_path / "coords_{domain}_{shape}.nc"),
        )

        maybe_unstacked = xs.utils.maybe_unstack(
            out,
            coords=str(tmp_path / "coords_{domain}_{shape}.nc"),
            stack_drop_nans=True,
        )
        unstacked = xs.utils.unstack_fill_nan(out, coords=str(tmp_path / "coords_{domain}_{shape}.nc"))

        assert maybe_unstacked.equals(unstacked)


class TestXclimConvertUnitsContext:
    def test_simple(self):
        pr = timeseries([0, 1, 2], variable="pr", start="2001-01-01", units="mm d-1")
        with xs.utils.xclim_convert_units_to():
            xsdba.units.convert_units_to(pr, "kg m-2 s-1")

    def test_functions_outside_units(self):
        pr_mm = timeseries([0, 1, 2], variable="pr", start="2001-01-01", units="mm d-1")
        pr_kg = timeseries([0, 1, 2], variable="pr", start="2001-01-01", units="kg m-2 s-1")
        with xs.utils.xclim_convert_units_to():
            xsdba.DetrendedQuantileMapping.train(pr_mm, pr_kg).ds.load()


class TestVariablesUnits:
    def test_variables_same(self):
        ds = timeseries(
            np.tile(np.arange(1, 13), 3),
            variable="tas",
            start="2001-01-01",
            freq="MS",
            as_dataset=True,
        )
        out = xs.clean_up(ds, variables_and_units={"tas": "degK"})
        assert out.tas.attrs["units"] == "degK"
        np.testing.assert_array_equal(out.tas, ds.tas)

        out2 = xs.clean_up(ds, variables_and_units={"tas": "°K"})
        assert out2.tas.attrs["units"] == "°K"

    def test_variables_2(self):
        ds = timeseries(
            np.tile(np.arange(1, 13), 3),
            variable="tas",
            start="2001-01-01",
            freq="MS",
            as_dataset=True,
        )
        ds["pr"] = timeseries(
            np.tile(np.arange(1, 13), 3),
            variable="pr",
            start="2001-01-01",
            freq="MS",
        )
        out = xs.clean_up(ds, variables_and_units={"tas": "degK"})
        assert out.tas.attrs["units"] == "degK"
        assert out.pr.attrs["units"] == "kg m-2 s-1"

    def test_variables_sametime(self):
        ds = timeseries(
            np.tile(np.arange(1, 13), 3),
            variable="pr",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
        )
        out = xs.clean_up(ds, variables_and_units={"pr": "mm/day"})
        assert out.pr.attrs["units"] == "mm/day"
        np.testing.assert_array_almost_equal(out.pr, ds.pr * 86400)

    def test_variables_amount2rate(self):
        ds = timeseries(
            np.tile(np.arange(1, 13), 3),
            variable="pr",
            start="2001-01-01",
            freq="D",
            units="mm",
            as_dataset=True,
        )
        out = xs.clean_up(ds, variables_and_units={"pr": "mm d-1"})
        assert out.pr.attrs["units"] == "mm d-1"
        np.testing.assert_array_almost_equal(out.pr, ds.pr)

    def test_variables_rate2amount(self):
        ds = timeseries(
            np.tile(np.arange(1, 13), 3),
            variable="pr",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
        )
        out = xs.clean_up(ds, variables_and_units={"pr": "mm"})
        assert out.pr.attrs["units"] == "mm"
        np.testing.assert_array_almost_equal(out.pr, ds.pr * 86400)

    def test_variables_error(self):
        ds = timeseries(
            np.tile(np.arange(1, 13), 3),
            variable="pr",
            start="2001-01-01",
            freq="D",
            units="mm s-2",
            as_dataset=True,
        )
        with pytest.raises(ValueError, match="No known transformation"):
            xs.clean_up(ds, variables_and_units={"pr": "mm"})


class TestCalendar:
    def test_normal(self):
        ds = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )

        out = xs.clean_up(ds, convert_calendar_kwargs={"calendar": "noleap"})
        assert isinstance(out.time.values[0], cftime.DatetimeNoLeap)
        assert len(out.time) == 365 * 4

    def test_360(self):
        ds = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )

        out = xs.clean_up(ds, convert_calendar_kwargs={"calendar": "360_day"})
        assert isinstance(out.time.values[0], cftime.Datetime360Day)
        assert len(out.time) == 360 * 4
        assert len(out.time.sel(time="2000-02-30")) == 1

    def test_missing_by_var(self):
        ds = datablock_3d(
            np.array(
                [
                    [np.arange(1, 365 * 4 + 2), np.arange(1, 365 * 4 + 2)],
                    [np.arange(1, 365 * 4 + 2), np.arange(1, 365 * 4 + 2)],
                ]
            ).T,
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
            x_start=0,
            x_step=1,
            x="rlon",
            y_start=0,
            y_step=1,
            y="rlat",
        )
        ds["pr"] = datablock_3d(
            np.array(
                [
                    [np.arange(1, 365 * 4 + 2), np.arange(1, 365 * 4 + 2)],
                    [np.arange(1, 365 * 4 + 2), np.arange(1, 365 * 4 + 2)],
                ]
            ).T,
            variable="pr",
            start="2000-01-01",
            freq="D",
            as_dataset=False,
            x_start=0,
            x_step=1,
            x="rlon",
            y_start=0,
            y_step=1,
            y="rlat",
        )
        ds = xs.clean_up(ds, convert_calendar_kwargs={"calendar": "noleap"})
        missing_by_vars = {"tas": "interpolate", "pr": 9999}

        out = xs.clean_up(
            ds,
            convert_calendar_kwargs={"calendar": "standard"},
            missing_by_var=missing_by_vars,
        )
        assert out.tas.isnull().sum() == 0
        np.testing.assert_array_equal(out.tas.sel(time="2000-02-29"), 60)
        assert out.pr.isnull().sum() == 0
        np.testing.assert_array_equal(out.pr.sel(time="2000-02-29"), 9999)
        assert ds.rlon.attrs["axis"] == "X"  # Check that the attributes are preserved
        assert ds.lon.attrs["units"] == "degrees_east"
        assert out.rlon.attrs["axis"] == "X"
        assert out.lon.attrs["units"] == "degrees_east"

    def test_missing_by_var_error(self):
        ds = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )
        ds["pr"] = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="pr",
            start="2000-01-01",
            freq="D",
        )
        missing_by_vars = {"pr": 9999}
        with pytest.raises(ValueError, match="All variables must be"):
            xs.clean_up(
                ds,
                convert_calendar_kwargs={"calendar": "standard"},
                missing_by_var=missing_by_vars,
            )

    def test_no_time(self):
        ds = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )
        ds["orog"] = ds["tas"].isel(time=0).drop_vars("time")
        out = xs.clean_up(
            ds,
            convert_calendar_kwargs={"calendar": "standard"},
        )
        assert "time" in out.tas.dims
        assert "time" not in out.orog.dims


def test_round():
    ds = timeseries(
        np.arange(1, 365 * 4 + 2) / 1234,
        variable="tas",
        start="2000-01-01",
        freq="D",
        as_dataset=True,
    )
    ds["pr"] = timeseries(
        np.arange(1, 365 * 4 + 2) / 1234,
        variable="pr",
        start="2000-01-01",
        freq="D",
    )
    out = xs.clean_up(ds, round_var={"tas": 6, "pr": 1})
    np.testing.assert_array_equal(out.tas.isel(time=0), 0.000810)
    np.testing.assert_array_equal(out.pr.isel(time=0), 0.0)
    assert "Rounded 'pr' to 1 decimal" in out["pr"].attrs["history"]


def test_clip():
    ds = timeseries(
        np.arange(1, 365),
        variable="hurs",
        start="2000-01-01",
        freq="D",
        as_dataset=True,
    )

    out = xs.clean_up(ds, clip_var={"hurs": [0, 100]})
    np.testing.assert_array_equal(out.hurs.isel(time=-1), 100)
    assert "Clipped 'hurs' to [0, 100]" in out["hurs"].attrs["history"]


class TestAttrs:
    def test_common(self):
        ds1 = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )

        ds2 = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )
        ds2.attrs = {
            "foo": "bar",
            "cat:type": "simulation",
            "cat:variable": ("tas",),
            "cat:source": "CNRM-CM6",
            "cat:mip_era": "CMIP6",
        }
        ds3 = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )
        ds3.attrs = {
            "foo": "bar",
            "cat:type": "simulation",
            "cat:variable": ("tas",),
            "cat:source": "CanESM5",
            "cat:mip_era": "CMIP6",
        }

        # Nothing in common between ds1 and the other datasets
        ds1.attrs = {"bar": "foo"}
        out = xs.clean_up(ds1, common_attrs_only=[ds2, ds3])
        assert out.attrs == {}

        ds1.attrs = ds2.attrs
        out = xs.clean_up(ds1, common_attrs_only=[ds2, ds3])
        assert all(k in out.attrs for k in ["foo", "cat:type", "cat:variable", "cat:mip_era", "cat:id"])
        assert out.attrs["cat:id"] == "CMIP6"

        del ds1.attrs["cat:mip_era"]
        out = xs.clean_up(ds1, common_attrs_only={"a": ds2, "b": ds3})
        assert all(k in out.attrs for k in ["foo", "cat:type", "cat:variable", "cat:id"])
        assert out.attrs["cat:id"] == ""

    @pytest.mark.requires_netcdf
    def test_common_open(self):
        ds1 = xr.open_dataset(
            Path(__file__).parent.parent
            / "docs"
            / "notebooks"
            / "samples"
            / "tutorial"
            / "ScenarioMIP"
            / "example-region"
            / "NCC"
            / "NorESM2-MM"
            / "ssp126"
            / "r1i1p1f1"
            / "day"
            / "ScenarioMIP_NCC_NorESM2-MM_ssp126_r1i1p1f1_gn_raw.nc"
        )
        ds1.attrs["cat:id"] = "SomeID"
        ds2 = (
            Path(__file__).parent.parent
            / "docs"
            / "notebooks"
            / "samples"
            / "tutorial"
            / "ScenarioMIP"
            / "example-region"
            / "NCC"
            / "NorESM2-MM"
            / "ssp245"
            / "r1i1p1f1"
            / "day"
            / "ScenarioMIP_NCC_NorESM2-MM_ssp245_r1i1p1f1_gn_raw.nc"
        )
        ds3 = (
            Path(__file__).parent.parent
            / "docs"
            / "notebooks"
            / "samples"
            / "tutorial"
            / "ScenarioMIP"
            / "example-region"
            / "NCC"
            / "NorESM2-MM"
            / "ssp585"
            / "r1i1p1f1"
            / "day"
            / "ScenarioMIP_NCC_NorESM2-MM_ssp585_r1i1p1f1_gn_raw.nc"
        )

        out = xs.clean_up(ds1, common_attrs_only=[ds2, ds3])
        assert out.attrs["comment"] == "This is a test file created for the xscen tutorial. This file is not a real CMIP6 file."
        assert out.attrs.get("cat:id") is None

    def test_to_level(self):
        ds = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )
        out = xs.clean_up(ds, to_level="cat")
        assert out.attrs["cat:processing_level"] == "cat"

    def test_remove(self):
        ds = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )
        ds.attrs = {
            "foo": "bar",
            "cat:type": "simulation",
            "cat:variable": ("tas",),
            "cat:source": "CNRM-CM6",
            "bacat:mip_era": "CMIP6",
        }
        out = xs.clean_up(ds, attrs_to_remove={"tas": ["units"], "global": [".*cat.*", "foo"]})
        assert "units" in ds.tas.attrs
        assert "units" not in out.tas.attrs
        assert out.attrs == {}
        out2 = xs.clean_up(ds, attrs_to_remove={"tas": ["units"], "global": ["cat.*"]})
        assert out2.attrs == {"foo": "bar", "bacat:mip_era": "CMIP6"}

    def test_remove_except(self):
        ds = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )
        ds.attrs = {
            "foo": "bar",
            "cat:type": "simulation",
            "cat:variable": ("tas",),
            "cat:source": "CNRM-CM6",
            "bacat:mip_era": "CMIP6",
        }
        out = xs.clean_up(ds, remove_all_attrs_except={"tas": ["units"], "global": [".*cat.*"]})
        assert out.tas.attrs == {"units": "K"}
        assert out.attrs == {
            "cat:type": "simulation",
            "cat:variable": ("tas",),
            "cat:source": "CNRM-CM6",
            "bacat:mip_era": "CMIP6",
        }
        out2 = xs.clean_up(ds, remove_all_attrs_except={"global": ["cat.*"]})
        assert len(out2.tas.attrs) == 4
        assert out2.attrs == {
            "cat:type": "simulation",
            "cat:variable": ("tas",),
            "cat:source": "CNRM-CM6",
        }

    def test_add_attr(self):
        ds = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )
        out = xs.clean_up(
            ds,
            add_attrs={"tas": {"foo": "bar"}, "global": {"foo2": "electric boogaloo"}},
        )
        assert out.tas.attrs["foo"] == "bar"
        assert out.attrs["foo2"] == "electric boogaloo"

    @pytest.mark.parametrize("change_prefix", ["dog", {"cat": "dog:"}, {"cat:": "dog:", "bacat": "badog"}])
    def test_change_prefix(self, change_prefix):
        ds = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )
        ds.attrs = {
            "foo": "bar",
            "cat:type": "simulation",
            "cat:variable": ("tas",),
            "cat:source": "CNRM-CM6",
            "bacat:mip_era": "CMIP6",
        }
        out = xs.clean_up(ds, change_attr_prefix=change_prefix)
        if isinstance(change_prefix, str) or len(change_prefix) == 1:
            assert out.attrs == {
                "foo": "bar",
                "dog:type": "simulation",
                "dog:variable": ("tas",),
                "dog:source": "CNRM-CM6",
                "bacat:mip_era": "CMIP6",
            }
        else:
            assert out.attrs == {
                "foo": "bar",
                "dog:type": "simulation",
                "dog:variable": ("tas",),
                "dog:source": "CNRM-CM6",
                "badog:mip_era": "CMIP6",
            }


class TestUnstackDates:
    @pytest.mark.parametrize("freq", ["MS", "2MS", "3MS", "QS-DEC", "QS", "2QS", "YS", "YS-DEC", "4YS"])
    def test_normal(self, freq):
        ds = timeseries(
            np.arange(1, 35),
            variable="tas",
            start="2000-01-01",
            freq=freq,
            as_dataset=True,
        )
        out = xs.utils.unstack_dates(ds)
        np.testing.assert_array_equal(
            out.time,
            pd.date_range(
                "2000-01-01",
                periods=len(np.unique(ds.time.dt.year)),
                freq="YS" if freq != "4YS" else "4YS",
            ),
        )
        if freq == "MS":
            np.testing.assert_array_equal(
                out.month,
                [
                    "JAN",
                    "FEB",
                    "MAR",
                    "APR",
                    "MAY",
                    "JUN",
                    "JUL",
                    "AUG",
                    "SEP",
                    "OCT",
                    "NOV",
                    "DEC",
                ],
            )
        elif "M" in freq:
            assert len(out.season) == 12 / int(freq[0])
            np.testing.assert_array_equal(out.season[0], ["JF"] if freq == "2MS" else ["JFM"])
        elif "QS" in freq:
            assert len(out.season) == 4 if freq != "2QS" else 2
            np.testing.assert_array_equal(
                out.season[0],
                (["MAM"] if freq == "QS-DEC" else ["JFM"] if freq == "QS" else ["JFMAMJ"]),
            )
        else:
            assert len(out.season) == 1
            np.testing.assert_array_equal(out.season[0], [f"{freq.replace('YS', 'annual')}"])

    @pytest.mark.parametrize("freq", ["MS", "QS", "YS"])
    def test_seasons(self, freq):
        ds = timeseries(
            np.arange(1, 35),
            variable="tas",
            start="2000-01-01",
            freq=freq,
            as_dataset=True,
        )
        seasons = {
            1: "january",
            2: "february",
            3: "march",
            4: "april",
            5: "may",
            6: "june",
            7: "july",
            8: "august",
            9: "september",
            10: "october",
            11: "november",
            12: "december",
        }
        out = xs.utils.unstack_dates(ds, seasons=seasons)
        if freq == "MS":
            np.testing.assert_array_equal(
                out.month,
                [
                    "january",
                    "february",
                    "march",
                    "april",
                    "may",
                    "june",
                    "july",
                    "august",
                    "september",
                    "october",
                    "november",
                    "december",
                ],
            )
        elif freq == "QS":
            np.testing.assert_array_equal(out.season, ["january", "april", "july", "october"])
        elif freq == "YS":
            np.testing.assert_array_equal(out.season, ["january"])

    @pytest.mark.parametrize("freq", ["2MS", "QS-DEC", "YS-DEC"])
    def test_winter(self, freq):
        ds = timeseries(
            np.arange(1, 35),
            variable="tas",
            start="2000-12-01",
            freq=freq,
            as_dataset=True,
        )
        out = xs.utils.unstack_dates(ds)
        assert pd.Timestamp(str(out.time[0].values).split("T")[0]) == pd.Timestamp("2000-01-01")
        out = xs.utils.unstack_dates(ds, winter_starts_year=True)
        assert pd.Timestamp(str(out.time[0].values).split("T")[0]) == pd.Timestamp("2001-01-01")

    def test_coords(self):
        freq = "MS"
        ds = timeseries(
            np.arange(1, 35),
            variable="tas",
            start="2000-01-01",
            freq=freq,
        )
        ds["horizon"] = xr.DataArray(
            np.array(["2001-2009"] * len(ds.time)),
            dims="time",
            coords={"time": ds.time},
        )
        ds = ds.assign_coords({"horizon": ds.horizon})
        out = xs.utils.unstack_dates(ds)
        assert all(k in out["horizon"].dims for k in ["time", "month"])

    def test_dask(self):
        freq = "MS"
        ds = timeseries(
            dask.array.from_array(np.arange(1, 35), chunks=(10,)),
            variable="tas",
            start="2000-01-01",
            freq=freq,
        )
        out = xs.utils.unstack_dates(ds)
        assert isinstance(out.data, dask.array.Array)
        assert "month" in out.dims

    def test_errors(self):
        ds = timeseries(
            np.arange(1, 365 * 4 + 2),
            variable="tas",
            start="2000-01-01",
            freq="D",
            as_dataset=True,
        )
        with pytest.raises(ValueError, match="Only monthly frequencies"):
            xs.utils.unstack_dates(ds)
        ds = ds.where(ds.time.dt.day != 1, drop=True)
        with pytest.raises(ValueError, match="The data must have a clean time coordinate."):
            xs.utils.unstack_dates(ds)

        ds = timeseries(
            np.arange(1, 13),
            variable="tas",
            start="2000-01-01",
            freq="7MS",
            as_dataset=True,
        )
        with pytest.raises(ValueError, match="Only periods that divide the year evenly are supported."):
            xs.utils.unstack_dates(ds)


class TestEnsureTime:
    def test_xrfreq_ok(self):
        ds = timeseries(
            np.arange(1, 360),
            variable="tas",
            start="2000-01-01T12:00:00",
            freq="D",
            as_dataset=True,
        )
        out = xs.utils.ensure_correct_time(ds, "D")
        assert np.all(out.time.dt.hour == 0)

    def test_xrfreq_bad(self):
        ds = timeseries(
            np.arange(1, 360),
            variable="tas",
            start="2000-01-01T12:00:00",
            freq="D",
            as_dataset=True,
        )
        # Add random small number of seconds to the time
        ds["time"] = ds.time + (np.random.rand(len(ds.time)) * 10).astype("timedelta64[s]")
        out = xs.utils.ensure_correct_time(ds, "D")
        assert np.all(out.time.dt.hour == 0)
        assert np.all(out.time.dt.second == 0)

    def test_xrfreq_error(self):
        ds = timeseries(
            np.arange(1, 360),
            variable="tas",
            start="2000-01-01T12:00:00",
            freq="D",
            as_dataset=True,
        )
        # Add random small number of seconds to the time
        rng = np.random.default_rng(0)
        ds["time"] = ds.time + (rng.random(len(ds.time)) * 24).astype("timedelta64[h]")
        with pytest.raises(
            ValueError,
            match="Dataset is labelled as having a sampling frequency of D, but some periods have more than one data point.",
        ):
            xs.utils.ensure_correct_time(ds, "D")
        ds = timeseries(
            np.arange(1, 360),
            variable="tas",
            start="2000-01-01T12:00:00",
            freq="D",
            as_dataset=True,
        )
        # Remove some time points
        ds = ds.where(ds.time.dt.day % 2 == 0, drop=True)
        with pytest.raises(
            ValueError,
            match="The resampling count contains NaNs or 0s. There might be some missing data.",
        ):
            xs.utils.ensure_correct_time(ds, "D")


class TestStandardPeriod:
    @pytest.mark.parametrize(
        "period, timestamp_eop_match, timestamp_neop_match",
        [
            ([1981, 2000], "2000-12-31 23:59:59", "2000-01-01"),
            ([[1981, 2000]], "2000-12-31 23:59:59", "2000-01-01"),
            (["1981", "2000"], "2000-12-31 23:59:59", "2000-01-01"),
            ([["1981", "2000-02"]], "2000-02-29 23:59:59", "2000-02-01"),
            ([["1981", "2000-12"]], "2000-12-31 23:59:59", "2000-12-01"),
            (
                [[pd.Timestamp("1981"), pd.Timestamp("2000")]],
                "2000-01-01 00:00:00",
                "2000-01-01",
            ),
        ],
    )
    def test_normal(self, period, timestamp_eop_match, timestamp_neop_match):
        out = xs.utils.standardize_periods(period, multiple=True)
        assert out == [["1981", "2000"]]

        out = xs.utils.standardize_periods(period, multiple=False)
        assert out == ["1981", "2000"]

        out = xs.utils.standardize_periods(period, multiple=False, out_dtype="datetime")
        assert out == [pd.Timestamp("1981-01-01"), pd.Timestamp(timestamp_eop_match)]

        out = xs.utils.standardize_periods(period, multiple=False, out_dtype="datetime", end_of_periods=False)
        assert out == [pd.Timestamp("1981-01-01"), pd.Timestamp(timestamp_neop_match)]

    def test_error(self):
        assert xs.utils.standardize_periods(None) is None
        with pytest.raises(ValueError, match="should be comprised of two elements"):
            xs.utils.standardize_periods(["1981-2010"])
        with pytest.raises(ValueError, match="should be in chronological order,"):
            xs.utils.standardize_periods(["2010", "1981"])
        with pytest.raises(ValueError, match="should be a single instance"):
            xs.utils.standardize_periods([["1981", "2010"], ["1981", "2010"]], multiple=False)


def test_sort_seasons():
    seasons = pd.Index(["JJA", "DJF", "SON", "MAM"])
    out = xs.utils.season_sort_key(seasons, name="season")
    np.testing.assert_array_equal(out, [6, 0, 9, 3])

    seasons = pd.Index(["JFM", "JAS", "OND", "AMJ"])
    out = xs.utils.season_sort_key(seasons, name="season")
    np.testing.assert_array_equal(out, [1, 7, 10, 4])

    seasons = pd.Index(["FEB", "JAN", "MAR", "DEC"])
    out = xs.utils.season_sort_key(seasons, name="month")
    np.testing.assert_array_equal(out, [1, 0, 2, 11])

    # Invalid returns the original object
    seasons = pd.Index(["FEB", "DEC", "foo"])
    out = xs.utils.season_sort_key(seasons, name="month")
    np.testing.assert_array_equal(out, seasons)


def test_xrfreq_to_timedelta():
    assert xs.utils.xrfreq_to_timedelta("D") == pd.Timedelta(1, "D")
    assert xs.utils.xrfreq_to_timedelta("QS-DEC") == pd.Timedelta(90, "D")
    assert xs.utils.xrfreq_to_timedelta("YS") == pd.Timedelta(365, "D")
    assert xs.utils.xrfreq_to_timedelta("2QS") == pd.Timedelta(180, "D")
    with pytest.raises(ValueError, match="Invalid frequency"):
        xs.utils.xrfreq_to_timedelta("foo")


def test_ensure_new_xrfreq():
    assert xs.utils.ensure_new_xrfreq("2M") == "2ME"
    assert xs.utils.ensure_new_xrfreq("2Q-DEC") == "2QE-DEC"
    assert xs.utils.ensure_new_xrfreq("AS-JUL") == "YS-JUL"
    assert xs.utils.ensure_new_xrfreq("A-JUL") == "YE-JUL"
    assert xs.utils.ensure_new_xrfreq("Y-JUL") == "YE-JUL"
    assert xs.utils.ensure_new_xrfreq("A") == "YE"
    assert xs.utils.ensure_new_xrfreq("Y") == "YE"
    assert xs.utils.ensure_new_xrfreq("3H") == "3h"
    assert xs.utils.ensure_new_xrfreq("3T") == "3min"
    assert xs.utils.ensure_new_xrfreq("3S") == "3s"
    assert xs.utils.ensure_new_xrfreq("3L") == "3ms"
    assert xs.utils.ensure_new_xrfreq("3U") == "3us"

    # Errors
    assert xs.utils.ensure_new_xrfreq(3) == 3
    assert xs.utils.ensure_new_xrfreq("foo") == "foo"


def test_xarray_defaults():
    kwargs = {
        "chunks": {"time": 10},
        "foo": "bar",
        "xr_open_kwargs": {"decode_times": False},
        "xr_combine_kwargs": {"combine_attrs": "drop"},
    }
    out = xs.utils._xarray_defaults(**kwargs)
    assert out == {
        "chunks": {"time": 10},
        "foo": "bar",
        "xarray_open_kwargs": {"decode_times": False, "chunks": {}},
        "xarray_combine_by_coords_kwargs": {
            "combine_attrs": "drop",
            "data_vars": "minimal",
        },
    }
