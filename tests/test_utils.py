from datetime import datetime

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs
from xscen.testing import datablock_3d


class TestDateParser:
    @pytest.mark.parametrize(
        "date,end_of_period,dtype,exp",
        [
            ("2001", True, "datetime", pd.Timestamp("2001-12-31 23:59:59")),
            ("150004", True, "datetime", pd.Timestamp("1500-04-30 23:59:59")),
            ("31231212", None, "datetime", pd.Timestamp("3123-12-12")),
            ("2001-07-08", None, "period", pd.Period("2001-07-08", "H")),
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
        ],
    )
    def test_normal(self, date, end_of_period, dtype, exp):
        out = xs.utils.date_parser(date, end_of_period=end_of_period, out_dtype=dtype)
        if pd.isna(exp):
            assert pd.isna(out)
        else:
            assert out == exp


class TestScripting:
    ds = timeseries(
        np.tile(np.arange(1, 2), 50),
        variable="tas",
        start="2000-01-01",
        freq="AS-JAN",
        as_dataset=True,
    )
    ds.attrs = {
        "cat:type": "simulation",
        "cat:processing_level": "raw",
        "cat:variable": ("tas",),
        "dog:source": "CanESM5",
    }

    @pytest.mark.parametrize(
        "prefix, var_as_str", [["cat:", False], ["cat:", True], ["dog:", True]]
    )
    def test_get_cat_attrs(self, prefix, var_as_str):
        out = xs.utils.get_cat_attrs(self.ds, prefix=prefix, var_as_str=var_as_str)

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


class TestStackNan:

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
            to_file=str(tmp_path / "coords_{domain}_{shape}.nc"),
        )
        assert "loc1" in out.dims
        assert out.sizes["loc1"] == 99
        assert (tmp_path / "coords_RegionEssai_10x10.nc").is_file()

        ds_unstack = xs.utils.unstack_fill_nan(
            out, dim="loc1", coords=str(tmp_path / "coords_{domain}_{shape}.nc")
        )
        assert ds_unstack.equals(ds)
