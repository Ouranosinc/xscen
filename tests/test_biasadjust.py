import numpy as np
import pytest
import xarray as xr
import xclim as xc
from conftest import notebooks
from xclim.sdba import stack_variables
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs

xc.set_options(
    sdba_encode_cf=False
)  # FIXME: A temporary bug fix waiting for xclim 0.49


class TestTrain:
    dref = timeseries(
        np.ones(365 * 3), variable="tas", start="2001-01-01", freq="D", as_dataset=True
    )

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
            adapt_freq={"thresh": "2 K"},
            jitter_over={"upper_bnd": "3 K", "thresh": "2 K"},
            jitter_under={"thresh": "2 K"},
        )

        assert out.attrs["train_params"] == {
            "maximal_calendar": "noleap",
            "adapt_freq": {"thresh": "2 K"},
            "jitter_over": {"upper_bnd": "3 K", "thresh": "2 K"},
            "jitter_under": {"thresh": "2 K"},
            "var": ["tas"],
        }

        assert "pth" in out
        assert "dP0" in out
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
        "periods, to_level, bias_adjust_institution, bias_adjust_project",
        [
            (["2001", "2006"], None, None, None),
            ([["2001", "2001"], ["2006", "2006"]], "test", "i", "p"),
        ],
    )
    def test_basic(
        self, periods, to_level, bias_adjust_institution, bias_adjust_project
    ):
        dtrain = xs.train(
            self.dref.copy(),
            self.dsim.sel(time=slice("2001", "2003")),
            var="tas",
            period=["2001", "2003"],
        )

        out = xs.adjust(
            dtrain,
            self.dsim.copy(),
            periods=periods,
            to_level=to_level,
            bias_adjust_institution=bias_adjust_institution,
            bias_adjust_project=bias_adjust_project,
        )
        assert out.attrs["cat:processing_level"] == to_level or "biasadjusted"
        assert out.attrs["cat:variable"] == ("tas",)
        assert out.attrs["cat:id"] == "fake_id"
        assert (
            out["tas"].attrs["bias_adjustment"]
            == "DetrendedQuantileMapping(group=Grouper("
            "name='time.dayofyear', window=31), kind='+'"
            ").adjust(sim, )"
        )
        assert out.time.dt.calendar == "noleap"

        if bias_adjust_institution is not None:
            assert out.attrs["cat:bias_adjust_institution"] == "i"
        if bias_adjust_project is not None:
            assert out.attrs["cat:bias_adjust_project"] == "p"

        assert out.time.dt.year.values[0] == 2001
        assert out.time.dt.year.values[-1] == 2006

        if periods == ["2001", "2006"]:
            np.testing.assert_array_equal(
                out["tas"].values,
                np.concatenate(
                    [np.ones(365 * 3) * 1, np.ones(365 * 3) * 3]
                ),  # -1 for leap year
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
            adapt_freq={"thresh": "2 K"},
            jitter_over={"upper_bnd": "3 K", "thresh": "2 K"},
            jitter_under={"thresh": "2 K"},
        )

        root = str(tmpdir / "_data")
        xs.save_to_zarr(dtrain, f"{root}/test.zarr", mode="o")
        dtrain2 = xr.open_dataset(
            f"{root}/test.zarr", chunks={"dayofyear": 365, "quantiles": 15}
        )

        out = xs.adjust(
            dtrain,
            self.dsim.copy(),
            periods=["2001", "2006"],
            xclim_adjust_args={
                "detrend": {
                    "LoessDetrend": {"f": 0.2, "niter": 1, "d": 0, "weights": "tricube"}
                }
            },
        )

        out2 = xs.adjust(
            dtrain2,
            self.dsim.copy(),
            periods=["2001", "2006"],
            xclim_adjust_args={
                "detrend": {
                    "LoessDetrend": {"f": 0.2, "niter": 1, "d": 0, "weights": "tricube"}
                }
            },
        )

        assert (
            out.tas.attrs["bias_adjustment"]
            == "DetrendedQuantileMapping(group=Grouper(name='time.dayofyear',"
            " window=31), kind='+').adjust(sim, detrend=<LoessDetrend>),"
            " ref and hist were prepared with jitter_under_thresh(ref, hist,"
            " {'thresh': '2 K'}) and jitter_over_thresh(ref, hist, {'upper_bnd':"
            " '3 K', 'thresh': '2 K'}) and adapt_freq(ref, hist, {'thresh': '2 K'})"
        )

        assert (
            out2.tas.attrs["bias_adjustment"]
            == "DetrendedQuantileMapping(group=Grouper(name='time.dayofyear',"
            " window=31), kind='+').adjust(sim, detrend=<LoessDetrend>), ref and"
            " hist were prepared with jitter_under_thresh(ref, hist, {'thresh':"
            " '2 K'}) and jitter_over_thresh(ref, hist, {'upper_bnd': '3 K',"
            " 'thresh': '2 K'}) and adapt_freq(ref, hist, {'thresh': '2 K'})"
        )

        assert out.equals(out2)

    def test_xclim_vs_xscen(
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
            adapt_freq={"thresh": "1 mm d-1"},
            xclim_train_args={"kind": "*", "nquantiles": 50},
        )

        out_xscen = xs.adjust(
            dtrain_xscen,
            dsim,
            periods=["2001", "2006"],
            xclim_adjust_args={
                "detrend": {
                    "LoessDetrend": {"f": 0.2, "niter": 1, "d": 0, "weights": "tricube"}
                },
                "interp": "nearest",
                "extrapolation": "constant",
            },
        )

        # xclim version
        with xc.set_options(sdba_extra_output=True):
            group = xc.sdba.Grouper(group="time.dayofyear", window=31)

            drefx = dref.sel(time=slice("2001", "2003")).convert_calendar("noleap")
            dhistx = dhist.sel(time=slice("2001", "2003")).convert_calendar("noleap")
            dsimx = dsim.sel(time=slice("2001", "2006")).convert_calendar("noleap")

            dhist_ad, pth, dP0 = xc.sdba.processing.adapt_freq(
                drefx["pr"], dhistx["pr"], group=group, thresh="1 mm d-1"
            )

            QM = xc.sdba.DetrendedQuantileMapping.train(
                drefx["pr"], dhist_ad, group=group, kind="*", nquantiles=50
            )

            detrend = xc.sdba.detrending.LoessDetrend(
                f=0.2, niter=1, d=0, weights="tricube", group=group, kind="*"
            )
            out_xclim = QM.adjust(
                dsimx["pr"], detrend=detrend, interp="nearest", extrapolation="constant"
            ).rename({"scen": "pr"})

        assert out_xscen.equals(out_xclim)

    def test_only_adjust(self):
        # that's an optional dependency in xclim, not ideal
        # but it's the only example of purely Adjust
        pytest.importorskip("ot")

        # dOTC fails with a uniform array like above, so this is needed
        dsim = timeseries(
            np.arange(365 * 3 + 1),
            variable="tas",
            start="2001-01-01",
            freq="D",
            as_dataset=True,
        )

        dsim.attrs["cat:xrfreq"] = "D"
        dsim.attrs["cat:domain"] = "one_point"
        dsim.attrs["cat:id"] = "fake_id"

        dsim = stack_variables(dsim).to_dataset()
        xs.adjust(
            dtrain=None,
            dsim=dsim,
            periods=["2003", "2005"],
            method="dOTC",
            xclim_adjust_args=dict(
                ref=dsim.sel(time=slice("2001", "2003")),
                hist=dsim.sel(time=slice("2001", "2003")),
            ),
        )
