from datetime import datetime

import cftime
import numpy as np
import pandas as pd
import pytest

import xscen as xs


class TestDateParser:
    @pytest.mark.parametrize(
        "date,end_of_period,dtype,exp",
        [
            ("2001", True, "datetime", pd.Timestamp("2001-12-31")),
            ("150004", True, "datetime", pd.Timestamp("1500-04-30")),
            ("31231212", None, "datetime", pd.Timestamp("3123-12-12")),
            ("2001-07-08", None, "period", pd.Period("2001-07-08", "H")),
            (pd.Timestamp("1993-05-20T12:07"), None, "str", "1993-05-20"),
            (
                cftime.Datetime360Day(1981, 2, 30),
                None,
                "datetime",
                pd.Timestamp("1981-02-28"),
            ),
            (np.datetime64("1200-11-12"), "Y", "datetime", pd.Timestamp("1200-12-31")),
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