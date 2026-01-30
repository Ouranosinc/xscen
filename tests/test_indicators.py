import warnings

import numpy as np
import pytest
import xclim
from conftest import notebooks
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs


class TestComputeIndicators:
    yaml_file = notebooks / "samples" / "indicators.yml"
    ds = timeseries(np.ones(365 * 3), variable="tas", start="2001-01-01", freq="D", as_dataset=True)

    @pytest.mark.parametrize("reload", [True, False])
    def test_reload_module(self, reload):
        module = xs.indicators.load_xclim_module(self.yaml_file)
        assert all(hasattr(module, ind) for ind in ["growing_degree_days", "tg_min"])

        # Record warnings without failing the test if no warnings are raised.
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            xs.indicators.load_xclim_module(notebooks / "samples" / "indicators", reload=reload)
        assert len([r for r in record if "already exists and will be overwritten." in str(r.message)]) == (2 if reload else 0)

    @pytest.mark.parametrize("input", ["module", "iter"])
    def test_input_types(self, input):
        module = xs.indicators.load_xclim_module(self.yaml_file)
        ind_dict1 = xs.compute_indicators(
            self.ds,
            indicators=module if input == "module" else module.iter_indicators(),
        )
        ind_dict2 = xs.compute_indicators(self.ds, indicators=self.yaml_file)
        assert all(ind_dict1[k].equals(ind_dict2[k]) for k in ind_dict1)

    @pytest.mark.parametrize("to_level", [None, "test"])
    def test_level(self, to_level):
        if to_level is None:
            ind_dict = xs.compute_indicators(self.ds, indicators=self.yaml_file)
            assert "indicators" in ind_dict["YS-JAN"].attrs["cat:processing_level"]
        else:
            ind_dict = xs.compute_indicators(self.ds, indicators=self.yaml_file, to_level=to_level)
            assert to_level in ind_dict["YS-JAN"].attrs["cat:processing_level"]

    # Periods needed to cover both branches of the if/else in compute_indicators, so no need to test separately.
    @pytest.mark.parametrize("periods", [None, [["2001", "2001"], ["2003", "2004"], ["2005", "2009"]]])
    def test_output(self, periods):
        values = np.ones(365 * (2 if periods is None else 10))
        ds = timeseries(values, variable="tas", start="2001-01-01", freq="D", as_dataset=True)
        ds["da"] = ds.tas

        module = xs.indicators.load_xclim_module(self.yaml_file)
        ind_dict = xs.compute_indicators(
            ds,
            indicators=[("fit", xclim.indicators.generic.fit)] + [x for x in module.iter_indicators()],
            periods=periods,
        )
        assert all(xrfreq in ind_dict[xrfreq].attrs["cat:xrfreq"] for xrfreq in ind_dict.keys())
        assert all(v in ind_dict["YS-JAN"] for v in ["tg_min", "growing_degree_days"])
        assert all(v in ind_dict["YS-JAN"].attrs["cat:variable"] for v in ["tg_min", "growing_degree_days"])
        if periods is None:
            assert "time" not in ind_dict["fx"].dims
            assert len(ind_dict["YS-JAN"].time) == 2
        else:
            assert len(ind_dict["fx"].time) == len(periods)
            assert len(ind_dict["YS-JAN"].time) == 8

    def test_qs_dec(self):
        indicator = xclim.core.indicator.Indicator.from_dict(
            data={"base": "tg_min", "parameters": {"freq": "QS-DEC"}},
            identifier="tg_min_qs",
            module="tests",
        )
        ind_dict = xs.compute_indicators(self.ds, indicators=[("tg_min_qs", indicator)])
        assert "QS-DEC" in ind_dict["QS-DEC"].attrs["cat:xrfreq"]
        assert len(ind_dict["QS-DEC"].time) == 12
        assert ind_dict["QS-DEC"].time[0].dt.strftime("%Y-%m-%d").item() == "2001-03-01"
        assert ind_dict["QS-DEC"].time[-1].dt.strftime("%Y-%m-%d").item() == "2003-12-01"

    # Periods needed to cover both branches of the if/else in compute_indicators, so no need to test separately.
    @pytest.mark.parametrize("periods", [None, [["2001", "2001"], ["2003", "2003"]]])
    def test_multiple_outputs(self, periods):
        values = np.zeros(365 * 3)
        values[150:250] = 100
        values[150 + 365 : 250 + 365] = 100
        values[150 + 365 * 2 : 250 + 365 * 2] = 100
        ds = timeseries(values, variable="pr", start="2001-01-01", freq="D", as_dataset=True)
        ind_dict = xs.compute_indicators(
            ds,
            indicators=[
                ("precip", xclim.atmos.precip_average),
                ("rs", xclim.atmos.rain_season),
            ],
            periods=periods,
        )

        assert all(
            v in ind_dict["YS-JAN"]
            for v in [
                "prcpavg",
                "rain_season_start",
                "rain_season_end",
                "rain_season_length",
            ]
        )

    @pytest.mark.parametrize("restrict_years", [True, False])
    def test_as_jul(self, restrict_years):
        indicator = xclim.core.indicator.Indicator.from_dict(
            data={"base": "freezing_degree_days", "parameters": {"freq": "YS-JUL"}},
            identifier="degree_days_below_0_annual_start_july",
            module="tests",
        )
        ind_dict = xs.compute_indicators(
            self.ds,
            indicators=[("degree_days_below_0_annual_start_july", indicator)],
            restrict_years=restrict_years,
        )

        assert "YS-JUL" in ind_dict["YS-JUL"].attrs["cat:xrfreq"]
        out = ind_dict["YS-JUL"]
        if restrict_years:
            assert len(out.time) == 3  # same length as input ds
            assert out.time[0].dt.strftime("%Y-%m-%d").item() == "2001-07-01"
            assert out.time[-1].dt.strftime("%Y-%m-%d").item() == "2003-07-01"
        else:
            assert len(out.time) == 4
            assert out.time[0].dt.strftime("%Y-%m-%d").item() == "2000-07-01"
            assert out.time[-1].dt.strftime("%Y-%m-%d").item() == "2003-07-01"

    @pytest.mark.parametrize("indicator_iter", ["list", "tuples", "module"])
    def test_select_inds_for_avail_vars(self, indicator_iter):
        # Test that select_inds_for_avail_vars filters a list of indicators to only those
        # that can be computed with the variables available in a dataset.
        ds = xs.testing.datablock_3d(
            np.ones((365, 3, 3)),
            variable="tas",
            x="lon",
            x_start=-75,
            y="lat",
            y_start=45,
            x_step=1,
            y_step=1,
            start="2001-01-01",
            freq="D",
            units="K",
            as_dataset=True,
        )
        indicators = [
            xclim.core.indicator.Indicator.from_dict(
                data={"base": "tg_min", "parameters": {"freq": "QS-DEC"}},
                identifier="tg_min_qs",
                module="tests",
            ),
            xclim.core.indicator.Indicator.from_dict(
                data={"base": "days_over_precip_thresh", "parameters": {"freq": "MS"}},
                identifier="precip_average_ms",
                module="tests",
            ),
        ]

        # indicators as different types
        module = xclim.core.indicator.build_indicator_module("indicators", {i.base: i for i in indicators}, reload=True)
        if indicator_iter == "list":
            inds_for_avail_vars = xs.indicators.select_inds_for_avail_vars(ds=ds, indicators=indicators)
        elif indicator_iter == "tuples":
            inds_for_avail_vars = xs.indicators.select_inds_for_avail_vars(ds=ds, indicators=[(n, i) for n, i in module.iter_indicators()])
        elif indicator_iter == "module":
            inds_for_avail_vars = xs.indicators.select_inds_for_avail_vars(ds=ds, indicators=module)

        assert len(list(inds_for_avail_vars.iter_indicators())) == 1
        assert [n for n, _ in inds_for_avail_vars.iter_indicators()] == ["tg_min"]
        assert [i for _, i in inds_for_avail_vars.iter_indicators()] == [indicators[0]]
        # no indicators found
        inds_for_avail_vars = xs.indicators.select_inds_for_avail_vars(ds=ds, indicators=indicators[1:])
        assert len(list(inds_for_avail_vars.iter_indicators())) == 0
        assert [(n, i) for n, i in inds_for_avail_vars.iter_indicators()] == []


@pytest.mark.parametrize(
    "ind,expvars,expfrq",
    [
        ("wind_vector_from_speed", ["uas", "vas"], "D"),
        ("fit", ["params"], "fx"),
        ("tg_mean", ["tg_mean"], "YS-JAN"),
    ],
)
def test_get_indicator_outputs(ind, expvars, expfrq):
    ind = xclim.core.indicator.registry[ind.upper()].get_instance()
    outvars, outfrq = xs.indicators.get_indicator_outputs(ind, "D")
    assert outvars == expvars
    assert outfrq == expfrq
