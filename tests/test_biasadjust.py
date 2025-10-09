import ast
import contextlib
import warnings

import numpy as np
import pytest
import xarray as xr
import xsdba
from xclim.testing.helpers import test_timeseries as timeseries

from xscen.utils import xclim_convert_units_to


# Copied from xarray/core/nputils.py
# Can be removed once numpy 2.0+ is the oldest supported version
try:
    from numpy.exceptions import RankWarning
except ImportError:
    from numpy import RankWarning

import xscen as xs


class TestTrain:
    dref = timeseries(np.ones(365 * 3), variable="tas", start="2001-01-01", freq="D", as_dataset=True)

    dhist = timeseries(
        np.concatenate([np.ones(365 * 2) * 2, np.ones(365) * 3]),
        variable="tas",
        start="2001-01-01",
        freq="D",
        as_dataset=True,
    )
    dhist.attrs["cat:xrfreq"] = "D"
    dhist.attrs["cat:domain"] = "one_point"
    dhist.attrs["cat:id"] = "fake_id"

    @pytest.mark.parametrize(
        "var, period",
        [("tas", ["2001", "2002"]), (["tas"], ["2003", "2003"])],
    )
    def test_basic_train(self, var, period):
        out = xs.train(self.dref, self.dhist, var=var, period=period)

        assert out.attrs["cat:xrfreq"] == "D"
        assert out.attrs["cat:domain"] == "one_point"
        assert out.attrs["cat:id"] == "fake_id"
        assert out.attrs["cat:processing_level"] == "training_tas"

        assert "dayofyear" in out
        assert "quantiles" in out
        result = [-1] * 365 if period == ["2001", "2002"] else [-2] * 365
        np.testing.assert_array_equal(out["scaling"], result)

    def test_preprocess(self):
        # FIXME: put back the test when xclim 0.55 is released, https://github.com/Ouranosinc/xclim/pull/2038/files
        # dhist360 = self.dhist.convert_calendar("360_day", align_on="year")
        dhist360 = self.dhist.convert_calendar("noleap", align_on="year")

        out = xs.train(
            self.dref,
            dhist360,
            var="tas",
            period=["2001", "2002"],
            xsdba_train_args=dict(adapt_freq_thresh="2 K"),
            jitter_over={"upper_bnd": "3 K", "thresh": "2 K"},
            jitter_under={"thresh": "2 K"},
        )

        assert out.attrs["train_params"] == {
            "maximal_calendar": "noleap",
            "jitter_over": {"upper_bnd": "3 K", "thresh": "2 K"},
            "jitter_under": {"thresh": "2 K"},
            "period": ["2001", "2002"],
            "var": ["tas"],
            "xsdba_train_args": {"adapt_freq_thresh": "2 K"},
            "additive_space": {},
        }

        assert "dayofyear" in out
        assert "quantiles" in out

    def test_group(self):
        out = xs.train(
            self.dref,
            self.dhist,
            var="tas",
            period=["2001", "2002"],
            group={"group": "time.month", "window": 1},
        )

        out1 = xs.train(
            self.dref,
            self.dhist,
            var="tas",
            period=["2001", "2002"],
            group="time.month",
        )

        assert "month" in out
        assert "quantiles" in out
        assert out1.equals(out)

    def test_errors(self):
        with pytest.raises(ValueError):
            xs.train(self.dref, self.dhist, var=["tas", "pr"], period=["2001", "2002"])


class TestAdjust:
    dref = timeseries(
        np.ones((365 * 3) + 1),  # leap year
        variable="tas",
        start="2001-01-01",
        freq="D",
        as_dataset=True,
    )

    dsim = timeseries(
        np.concatenate([np.ones(365 * 3) * 2, np.ones((365 * 3) + 1) * 4]),
        variable="tas",
        start="2001-01-01",
        freq="D",
        as_dataset=True,
    )
    dsim.attrs["cat:xrfreq"] = "D"
    dsim.attrs["cat:domain"] = "one_point"
    dsim.attrs["cat:id"] = "fake_id"

    @pytest.mark.parametrize(
        "periods, to_level, bias_adjust_institution, bias_adjust_project, bias_adjust_reference",
        [
            (["2001", "2006"], None, None, None, None),
            ([["2001", "2001"], ["2006", "2006"]], "test", "i", "p", "r"),
        ],
    )
    def test_basic(
        self,
        periods,
        to_level,
        bias_adjust_institution,
        bias_adjust_project,
        bias_adjust_reference,
    ):
        dtrain = xs.train(
            self.dref.copy(),
            self.dsim.sel(time=slice("2001", "2003")),
            var="tas",
            period=["2001", "2003"],
        )

        # For justification of warning filter, see: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.polyfit.html
        if to_level is None and bias_adjust_institution is None and bias_adjust_project is None and bias_adjust_reference is None:
            # No warning is expected
            context = contextlib.nullcontext()
        else:
            context = pytest.warns(RankWarning)
        with context:
            out = xs.adjust(
                dtrain,
                self.dsim.copy(),
                periods=periods,
                to_level=to_level,
                bias_adjust_institution=bias_adjust_institution,
                bias_adjust_project=bias_adjust_project,
                bias_adjust_reference=bias_adjust_reference,
            )

        assert out.attrs["cat:processing_level"] == to_level or "biasadjusted"
        assert out.attrs["cat:variable"] == ("tas",)
        assert out.attrs["cat:id"] == "fake_id"
        assert (
            out["tas"].attrs["bias_adjustment"] == "DetrendedQuantileMapping(group=Grouper("
            "name='time.dayofyear', window=31), kind='+'"
            ", adapt_freq_thresh=None).adjust(sim, ) with xsdba_train_args: {}"
        )
        assert out.time.dt.calendar == "noleap"

        if bias_adjust_institution is not None:
            assert out.attrs["cat:bias_adjust_institution"] == "i"
        if bias_adjust_project is not None:
            assert out.attrs["cat:bias_adjust_project"] == "p"
        if bias_adjust_reference is not None:
            assert out.attrs["cat:bias_adjust_reference"] == "r"

        assert out.time.dt.year.values[0] == 2001
        assert out.time.dt.year.values[-1] == 2006

        if periods == ["2001", "2006"]:
            np.testing.assert_array_equal(
                out["tas"].values,
                np.concatenate([np.ones(365 * 3) * 1, np.ones(365 * 3) * 3]),  # -1 for leap year
            )
        else:  # periods==[['2001','2001'], ['2006','2006']]
            np.testing.assert_array_equal(
                out["tas"].values,
                np.concatenate([np.ones(365 * 1) * 1, np.ones(365 * 1) * 3]),
            )

    def test_write_train(self, tmpdir):
        dtrain = xs.train(
            self.dref.copy(),
            self.dsim.sel(time=slice("2001", "2003")),
            var="tas",
            period=["2001", "2003"],
            xsdba_train_args=dict(adapt_freq_thresh="2 K"),
            jitter_over={"upper_bnd": "3 K", "thresh": "2 K"},
            jitter_under={"thresh": "2 K"},
        )

        root = str(tmpdir / "_data")
        xs.save_to_zarr(dtrain, f"{root}/test.zarr", mode="o")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dtrain2 = xr.open_dataset(f"{root}/test.zarr", chunks={"dayofyear": 365, "quantiles": 15})

        out = xs.adjust(
            dtrain,
            self.dsim.copy(),
            periods=["2001", "2006"],
            xsdba_adjust_args={"detrend": {"LoessDetrend": {"f": 0.2, "niter": 1, "d": 0, "weights": "tricube"}}},
        )

        out2 = xs.adjust(
            dtrain2,
            self.dsim.copy(),
            periods=["2001", "2006"],
            xsdba_adjust_args={"detrend": {"LoessDetrend": {"f": 0.2, "niter": 1, "d": 0, "weights": "tricube"}}},
        )

        assert (
            out.tas.attrs["bias_adjustment"] == "DetrendedQuantileMapping(group=Grouper(name='time.dayofyear',"
            " window=31), kind='+', adapt_freq_thresh='2 K').adjust(sim, detrend=<LoessDetrend>)"
            " with xsdba_train_args: {'adapt_freq_thresh': '2 K'},"
            " ref and hist were prepared with jitter_under_thresh(ref, hist,"
            " {'thresh': '2 K'}) and jitter_over_thresh(ref, hist, {'upper_bnd':"
            " '3 K', 'thresh': '2 K'})"
        )

        assert (
            out2.tas.attrs["bias_adjustment"] == "DetrendedQuantileMapping(group=Grouper(name='time.dayofyear',"
            " window=31), kind='+', adapt_freq_thresh='2 K').adjust(sim, detrend=<LoessDetrend>)"
            " with xsdba_train_args: {'adapt_freq_thresh': '2 K'}, "
            "ref and hist were prepared with jitter_under_thresh(ref, hist, {'thresh':"
            " '2 K'}) and jitter_over_thresh(ref, hist, {'upper_bnd': '3 K',"
            " 'thresh': '2 K'})"
        )

        assert out.equals(out2)

    def test_write_train_mbcn(self, tmpdir):
        tasmax = timeseries(
            np.arange(365 * 60),
            variable="tasmax",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
            calendar="noleap",
        )
        tasmin = timeseries(
            np.arange(365 * 60) / 2,
            variable="tasmin",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
            calendar="noleap",
        )
        ds = xr.merge([tasmax, tasmin])
        dtrain = xs.train(
            ds,
            ds,
            var=["tasmax", "tasmin"],
            method="MBCn",
            period=["2001", "2030"],
            xsdba_train_args={"base_kws": {"group": "time", "nquantiles": 15}},
        )

        root = str(tmpdir / "_data")
        xs.save_to_zarr(dtrain, f"{root}/test.zarr", mode="o")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            xr.open_dataset(f"{root}/test.zarr")

        params = {
            "var": ["tasmax", "tasmin"],
            "maximal_calendar": "noleap",
            "xsdba_train_args": {"base_kws": {"group": "time", "nquantiles": 15}},
            "jitter_under": None,
            "jitter_over": None,
            "period": ["2001", "2030"],
            "additive_space": {},
        }

        assert ast.literal_eval(dtrain.attrs["train_params"]) == params

        xs.adjust(
            dtrain=dtrain,
            dsim=ds,
            periods=["2031", "2060"],
            dref=ds,
        )

    def test_xsdba_vs_xscen(
        self,
    ):  # should give the same results  using xscen and xclim
        dref = (
            timeseries(
                np.random.randint(0, high=10, size=(365 * 3) + 1),
                variable="pr",
                start="2001-01-01",
                freq="D",
                as_dataset=True,
            )
            .astype("float32")
            .chunk({"time": -1})
        )

        dsim = (
            timeseries(
                np.random.randint(0, high=10, size=365 * 6 + 1),
                variable="pr",
                start="2001-01-01",
                freq="D",
                as_dataset=True,
            )
            .astype("float32")
            .chunk({"time": -1})
        )
        dhist = dsim.sel(time=slice("2001", "2003")).chunk({"time": -1})

        # xscen version
        dtrain_xscen = xs.train(
            dref,
            dhist,
            var="pr",
            period=["2001", "2003"],
            xsdba_train_args={
                "kind": "*",
                "nquantiles": 50,
                # FIXME: when xsdba can handle mm/d correctly
                # FIXME: Can't use adapt_freq_thresh because randomness is involved
                # "adapt_freq_thresh": "1.157e-05 kg/m2/s",
            },
        )

        out_xscen = xs.adjust(
            dtrain_xscen,
            dsim,
            periods=["2001", "2006"],
            xsdba_adjust_args={
                "detrend": {"LoessDetrend": {"f": 0.2, "niter": 1, "d": 0, "weights": "tricube"}},
                "interp": "nearest",
                "extrapolation": "constant",
            },
        )

        # xsdba version
        with xsdba.set_options(extra_output=True):
            group = xsdba.Grouper(group="time.dayofyear", window=31)

            drefx = dref.sel(time=slice("2001", "2003")).convert_calendar("noleap")
            dhistx = dhist.sel(time=slice("2001", "2003")).convert_calendar("noleap")
            dsimx = dsim.sel(time=slice("2001", "2006")).convert_calendar("noleap")

            # xsdba is now climate-agnostic, patch xclim's converter
            with xclim_convert_units_to():
                QM = xsdba.DetrendedQuantileMapping.train(
                    drefx["pr"],
                    dhistx["pr"],
                    group=group,
                    kind="*",
                    nquantiles=50,
                    # adapt_freq_thresh="1.157e-05 kg/m2/s",
                )

                detrend = xsdba.detrending.LoessDetrend(f=0.2, niter=1, d=0, weights="tricube", group=group, kind="*")
                out_xclim = QM.adjust(
                    dsimx["pr"],
                    detrend=detrend,
                    interp="nearest",
                    extrapolation="constant",
                ).rename({"scen": "pr"})

        xr.testing.assert_equal(out_xscen, out_xclim)

    def test_additive_space(self):
        data = np.random.random(365 * 3) * 90
        dhist = timeseries(data, variable="hurs", start="2001-01-01", freq="D", as_dataset=True)
        data = np.random.random(365 * 3) * 100

        dref = timeseries(data, variable="hurs", start="2001-01-01", freq="D", as_dataset=True)
        data2 = np.random.random(365 * 6) * 100
        dsim = timeseries(data2, variable="hurs", start="2001-01-01", freq="D", as_dataset=True)

        dtrain = xs.train(
            dref,
            dhist,
            var="hurs",
            period=["2001", "2003"],
            group="time",
            xsdba_train_args={"kind": "+"},
            additive_space={"hurs": dict(lower_bound="0 %", upper_bound="100 %", trans="logit")},
        )

        assert dtrain.attrs["train_params"]["additive_space"] == {"hurs": {"lower_bound": "0 %", "upper_bound": "100 %", "trans": "logit"}}

        dadjust = xs.adjust(dtrain, dsim, periods=["2001", "2007"])

        assert dadjust.hurs.max().values <= 100


class TestMultivariate:
    def test_mbcn(self):
        tasmax = timeseries(
            np.arange(365 * 60),
            variable="tasmax",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
            calendar="noleap",
        )
        tasmin = timeseries(
            np.arange(365 * 60) / 2,
            variable="tasmin",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
            calendar="noleap",
        )
        ds = xr.merge([tasmax, tasmin])
        dtrain = xs.train(
            ds,
            ds,
            var=["tasmax", "tasmin"],
            method="MBCn",
            period=["2001", "2030"],
            xsdba_train_args={"base_kws": {"group": "time"}},
        )
        xs.adjust(
            dtrain=dtrain,
            dsim=ds,
            periods=[
                ["2001", "2030"],
                ["2031", "2060"],
            ],
            dref=ds,
        )

    def test_stacking(self):
        tasmax = timeseries(
            np.arange(365 * 60),
            variable="tasmax",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
            calendar="noleap",
        )
        tasmin = timeseries(
            np.arange(365 * 60) / 2,
            variable="tasmin",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
            calendar="noleap",
        )
        ds = xr.merge([tasmax, tasmin])
        # define a slightly different dataset
        np.random.seed(42)
        dtx = np.random.rand(tasmax.time.size)
        dtn = np.random.rand(tasmin.time.size)
        with xr.set_options(keep_attrs=True):
            ds2 = xr.merge([tasmax + dtx, tasmin + dtn])
        dtrain = xs.train(
            ds,
            ds2,
            var=["tasmax", "tasmin"],
            method="MBCn",
            period=["2001", "2030"],
            xsdba_train_args={"base_kws": {"group": "time"}},
        )
        dscen_looped = xs.adjust(
            dtrain=dtrain,
            dsim=ds2,
            periods=[
                ["2001", "2030"],
                ["2031", "2060"],
            ],
            dref=ds,
        )
        dscen_stacked = xs.adjust(
            dtrain=dtrain,
            dsim=ds2,
            periods=["2001", "2060"],
            xsdba_adjust_args={"period_dim": "period"},
            dref=ds,
        )
        np.testing.assert_array_equal(dscen_looped.tasmax.values, dscen_stacked.tasmax.values)
