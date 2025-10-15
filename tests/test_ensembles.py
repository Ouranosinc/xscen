import logging
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr


try:
    import xesmf as xe
except (ImportError, KeyError) as e:
    if isinstance(e, KeyError):
        if e.args[0] != "Author":
            raise e
    xe = None

from xclim.testing.helpers import test_timeseries as timeseries
from xclim.testing.utils import nimbus

import xscen as xs


LOGGER = logging.getLogger(__name__)


class TestEnsembleStats:
    @staticmethod
    def make_ensemble(n):
        ens = []
        for i in range(n):
            tmp = timeseries(
                np.arange(1, 5) + i / 10,
                variable="tas",
                start="1981-01-01",
                freq="30YS",
                as_dataset=True,
            )
            tmp = tmp.assign_coords(
                {
                    "horizon": xr.DataArray(
                        ["1981-2010", "2011-2040", "2041-2070", "2071-2100"],
                        dims="time",
                    )
                }
            )
            tmp = tmp.rename({"tas": "tg_mean"})

            ens.append(tmp)

        return ens

    def test_input_type(self, tmpdir):
        ens_dict = self.make_ensemble(10)
        out_dict = xs.ensemble_stats(ens_dict, statistics={"ensemble_mean_std_max_min": None})
        ens_ds = xr.concat(ens_dict, dim="realization")
        out_ds = xs.ensemble_stats(ens_ds, statistics={"ensemble_mean_std_max_min": None})
        paths = []
        for i, ds in enumerate(ens_dict):
            paths.append(tmpdir / f"ens{i}.zarr")
            ds.to_zarr(paths[-1])
        out_zarr = xs.ensemble_stats(paths, statistics={"ensemble_mean_std_max_min": None}).compute()

        assert out_dict.equals(out_ds)
        # The zarr introduced some rounding errors
        assert np.round(out_zarr, 10).equals(np.round(out_dict, 10))

    @pytest.mark.parametrize("weights, to_level", [(None, None), (np.arange(1, 11), "for_tests")])
    def test_weights(self, weights, to_level):
        ens = self.make_ensemble(10)
        if weights is not None:
            weights = xr.DataArray(weights, dims="realization", coords={"realization": np.arange(0, 10)})

        if to_level is None:
            out = xs.ensemble_stats(
                ens,
                statistics={
                    "ensemble_mean_std_max_min": None,
                    "ensemble_percentiles": {"split": False},
                },
                weights=weights,
            )
        else:
            out = xs.ensemble_stats(
                ens,
                statistics={
                    "ensemble_mean_std_max_min": None,
                    "ensemble_percentiles": {"split": False},
                },
                weights=weights,
                to_level=to_level,
            )

        # Check that the output has the right attributes
        assert out.attrs["ensemble_size"] == 10
        assert out.attrs["cat:processing_level"] == to_level if to_level is not None else "ensemble"
        assert "realization" not in out.dims
        # Check that the output has the right variables and results
        np.testing.assert_array_equal(
            out.tg_mean_mean,
            [1.45, 2.45, 3.45, 4.45] if weights is None else [1.6, 2.6, 3.6, 4.6],
        )
        np.testing.assert_array_almost_equal(
            out.tg_mean_stdev,
            [0.2872] * 4 if weights is None else [0.2449] * 4,
            decimal=4,
        )
        np.testing.assert_array_almost_equal(
            out.tg_mean.sel(percentiles=90),
            [1.81, 2.81, 3.81, 4.81] if weights is None else [1.87, 2.87, 3.87, 4.87],
            decimal=2,
        )

    @pytest.mark.parametrize("fractions", ["only", "both", "nested", "missing"])
    def test_robustness_input(self, fractions):
        ens = self.make_ensemble(10)

        weights = None
        if fractions == "only":
            statistics = {"robustness_fractions": {"test": "threshold", "abs_thresh": 2.5}}
        elif fractions == "both":
            statistics = {
                "robustness_fractions": {"test": "threshold", "abs_thresh": 2.5},
                "robustness_categories": {
                    "categories": ["robust", "non-robust"],
                    "thresholds": [(0.5, 0.5), (0.5, 0.5)],
                    "ops": [(">=", ">="), ("<", "<")],
                },
            }
        elif fractions == "nested":
            weights = xr.DataArray([0] * 5 + [1] * 5, dims="realization")  # Mostly to check that this is put in the nested dict
            statistics = {
                "robustness_categories": {
                    "robustness_fractions": {"test": "threshold", "abs_thresh": 2.5},
                    "categories": ["robust", "non-robust"],
                    "thresholds": [(0.5, 0.5), (0.5, 0.5)],
                    "ops": [(">=", ">="), ("<", "<")],
                }
            }
        elif fractions == "missing":
            statistics = {
                "robustness_categories": {
                    "categories": ["robust", "non-robust"],
                    "thresholds": [(0.5, 0.5), (0.5, 0.5)],
                    "ops": [(">=", ">="), ("<", "<")],
                }
            }
            with pytest.raises(
                ValueError,
                match="'robustness_categories' requires 'robustness_fractions'",
            ):
                xs.ensemble_stats(ens, statistics=statistics)
            return

        out = xs.ensemble_stats(ens, statistics=statistics, weights=weights)

        # Output length should be 1 if nested, >=5 otherwise
        if fractions == "nested":
            assert len(out.data_vars) == 1
        else:
            assert len(out.data_vars) >= 5
        if fractions in ["only", "both"]:
            np.testing.assert_array_equal(out.tg_mean_changed, [0, 0.4, 1, 1])
            np.testing.assert_array_equal(out.tg_mean_agree, [1, 1, 1, 1])
        if fractions in ["both", "nested"]:
            np.testing.assert_array_equal(
                out.tg_mean_robustness_categories,
                {"both": [99, 99, 1, 1], "nested": [99, 1, 1, 1]}[fractions],
            )
            np.testing.assert_array_equal(
                out.tg_mean_robustness_categories.attrs["flag_descriptions"],
                ["robust", "non-robust"],
            )

    @pytest.mark.parametrize("symbol", ["rel.", "relative", "*", "/", "pct.", "abs."])
    def test_robustness_reldelta(self, caplog, symbol):
        ens = self.make_ensemble(10)
        for e in ens:
            e["tg_mean"].attrs["delta_kind"] = symbol

        with caplog.at_level(logging.INFO):
            xs.ensemble_stats(
                ens,
                statistics={"robustness_fractions": {"test": "threshold", "abs_thresh": 2.5}},
            )
        if symbol in ["rel.", "relative", "*", "/"]:
            assert "Relative delta detected" in caplog.text
        else:
            assert "Relative delta detected" not in caplog.text

    @pytest.mark.parametrize("common_attrs_only", [True, False])
    def test_common_attrs_only(self, common_attrs_only):
        ens = self.make_ensemble(10)
        for e in range(len(ens)):
            ens[e].attrs["foo"] = "bar"
            ens[e].attrs[f"bar{e}"] = "foo"

        out = xs.ensemble_stats(
            ens,
            statistics={"ensemble_mean_std_max_min": None},
            common_attrs_only=common_attrs_only,
        )
        assert out.attrs.get("foo", None) == "bar"
        assert ("bar0" not in out.attrs) == common_attrs_only

    def test_errors(self):
        ens = self.make_ensemble(10)

        # Warning if the statistic does not support weighting
        weights = xr.DataArray([0] * 5 + [1] * 5, dims="realization")
        with pytest.warns(UserWarning, match="Weighting is not supported"):
            with pytest.raises(TypeError):  # kkz is not actually supported here, but it's one of the few that will not support weighting
                xs.ensemble_stats(ens, statistics={"kkz_reduce_ensemble": None}, weights=weights)

        # Error if you try to use a relative delta with a reference dataset
        for e in ens:
            e["tg_mean"].attrs["delta_kind"] = "rel."
        ref = weights
        with pytest.raises(ValueError, match="is a delta, but 'ref' was still specified."):
            xs.ensemble_stats(
                ens,
                statistics={
                    "robustness_fractions": {
                        "test": "threshold",
                        "abs_thresh": 2.5,
                        "ref": ref,
                    }
                },
            )


class TestGenerateWeights:
    @staticmethod
    def make_ensemble():
        out = {}

        gcms = {
            "CCCma": ["CanESM2"],
            "GFDL": ["GFDL-ESM2M", "GFDL-ESM2G"],
            "CSIRO-QCCCE": ["CSIRO-Mk3-6-0"],
            "ECMWF": ["EC-EARTH"],
        }
        n_members = {
            "CanESM2": 3,
            "GFDL-ESM2M": 1,
            "GFDL-ESM2G": 2,
            "CSIRO-Mk3-6-0": 5,
            "EC-EARTH": 1,
        }
        rcms = {
            "CRCM5": ["CanESM2-r1i1p1"],
            "CanRCM4": ["CanESM2-r1i1p1", "CanESM2-r3i1p1"],
            "HIRHAM5": ["GFDL-ESM2M-r1i1p1", "EC-EARTH-r1i1p1"],
            "RegCM4": ["EC-EARTH-r1i1p1"],
        }

        for institution in ["CCCma", "GFDL", "CSIRO-QCCCE", "ECMWF"]:
            for gcm in gcms[institution]:
                members = [f"r{i}i1p1" for i in range(1, n_members[gcm] + 1)]
                for member in members:
                    has_gcm = [] if ((gcm == "CanESM2" and member == "r3i1p1") or gcm == "EC-EARTH") else [gcm]
                    rcm = [r for r in rcms if f"{gcm}-{member}" in rcms[r]]
                    for model in has_gcm + rcm:
                        ds = timeseries(
                            np.arange(1, 5),
                            variable="tas",
                            start="1981-01-01",
                            freq="30YS",
                            as_dataset=True,
                        )
                        ds["horizon"] = xr.DataArray(["1981-2010", "2041-2070", "+2C", "+4C"], dims="time")
                        ds = ds.swap_dims({"time": "horizon"}).drop_vars("time")
                        if f"{gcm}-{member}" == "CanESM2-r3i1p1":
                            ds = ds.where(ds.horizon == "1981-2010")
                        elif gcm == "GFDL-ESM2G":
                            ds = ds.where(~ds.horizon.str.startswith("+"))
                        elif gcm in ["EC-EARTH", "CSIRO-Mk3-6-0"]:
                            ds = ds.where(ds.horizon != "+4C")

                        ds.attrs = {
                            "cat:institution": institution,
                            "cat:member": member,
                            "cat:experiment": "rcp85" if gcm == "EC-EARTH" else "rcp45",
                        }
                        if model == gcm:
                            ds.attrs["cat:source"] = model
                            ds.attrs["cat:activity"] = "CMIP5"
                        else:
                            ds.attrs["cat:source"] = model
                            ds.attrs["cat:driving_model"] = gcm
                            ds.attrs["cat:activity"] = "CORDEX"

                        out[f"{institution}-{gcm}-{ds.attrs['cat:experiment']}-{member}-{model}"] = ds

                        if gcm == "CanESM2":
                            ds2 = ds.copy()
                            ds2.attrs["cat:experiment"] = "rcp85"
                            out[f"{institution}-{gcm}-{ds2.attrs['cat:experiment']}-{member}-{model}"] = ds2

                            if model == "CRCM5" and member == "r1i1p1":
                                for i in range(10):
                                    ds2 = ds.copy()
                                    ds2.attrs["cat:experiment"] = "rcp85"
                                    ds2.attrs["cat:member"] = f"xr{i}r1i1p1"
                                    ds2.attrs["cat:activity"] = "ClimEx"
                                    out[f"{institution}-{gcm}-{ds2.attrs['cat:experiment']}-{ds2.attrs['cat:member']}-{model}"] = ds2
                        elif gcm == "CSIRO-Mk3-6-0" and member == "r1i1p1":
                            ds2 = ds.copy()
                            ds2.attrs["cat:experiment"] = "rcp85"
                            ds2.attrs["cat:source"] = "CSIRO2"
                            out[f"{institution}-CSIRO2-{member}-{ds2.attrs['cat:experiment']}-CSIRO2"] = ds2
        return {o: out[o] for o in sorted(out)}

    @staticmethod
    def make_ensemble_rcm(ens):
        ens_rcm = ens.copy()
        for k in ens.keys():
            if "cat:driving_model" not in ens_rcm[k].attrs.keys():
                ens_rcm.pop(k)
        return ens_rcm

    ens = make_ensemble.__func__()
    ens_rcm = make_ensemble_rcm.__func__(ens)

    @staticmethod
    def make_answer(independence_level, exp_weights, skipna):
        answers = {
            "model-True": np.concatenate(
                (
                    np.array(
                        [
                            [0.14285, 0.14285, 0.16666, 0.2],
                            [0.07142, 0.07142, 0.08333, 0.1],
                            [0.07142, 0.14285, 0.16666, 0.2],
                            [0.07142, 0.07142, 0.08333, 0.1],
                            [0.07142, 0, 0, 0],
                        ]
                    ),  # CanESM2 family, RCP4.5
                    np.array(
                        [
                            [0.01515, 0.01515, 0.01515, 0.03030],
                            [0.08333, 0.08333, 0.08333, 0.16666],
                            [0.08333, 0.16666, 0.16666, 0.33333],
                            [0.08333, 0.08333, 0.08333, 0.16666],
                            [0.08333, 0, 0, 0],
                        ]
                    ),  # CanESM2 family, RCP8.5
                    np.repeat(np.array([[0.01515, 0.01515, 0.01515, 0.03030]]), 10, 0),  # ClimEx
                    np.repeat(np.array([[0.02857, 0.02857, 0.03333, 0]]), 5, 0),  # CSIRO-Mk6, RCP4.5
                    np.array([[0.16666, 0.16666, 0.16666, 0]]),  # CSIRO2, RCP8.5
                    np.array([[0.16666, 0.16666, 0.16666, 0], [0.16666, 0.16666, 0.16666, 0]]),  # EC-EARTH RCMs, RCP8.5
                    np.array([[0.07142, 0.07142, 0, 0], [0.07142, 0.07142, 0, 0]]),  # GFDL-ESM2G family, RCP4.5
                    np.array(
                        [
                            [0.14285, 0.14285, 0.16666, 0.2],
                            [0.14285, 0.14285, 0.16666, 0.2],
                        ]
                    ),  # GFDL-ESM2M family, RCP4.5
                )
            ),
            "model-False": np.concatenate(
                (
                    np.array(
                        [
                            [0.08333] * 4,
                            [0.25] * 4,
                            [0.25] * 1 + [0.5] * 3,
                            [0.25] * 4,
                            [0.25] * 1 + [0] * 3,
                        ]
                    ),  # CanESM2 family, RCP4.5
                    np.array(
                        [
                            [0.08333] * 4,
                            [0.25] * 4,
                            [0.25] * 1 + [0.5] * 3,
                            [0.25] * 4,
                            [0.25] * 1 + [0] * 3,
                        ]
                    ),  # CanESM2 family, RCP8.5
                    np.repeat(np.array([[0.08333] * 4]), 10, 0),  # ClimEx
                    np.repeat(np.array([[0.2] * 3 + [0] * 1]), 5, 0),  # CSIRO-Mk6, RCP4.5
                    np.array([[1] * 3 + [0] * 1]),  # CSIRO2, RCP8.5
                    np.array([[1] * 3 + [0] * 1, [1] * 3 + [0] * 1]),  # EC-EARTH RCMs, RCP8.5
                    np.array([[0.5] * 2 + [0] * 2, [0.5] * 2 + [0] * 2]),  # GFDL-ESM2G family, RCP4.5
                    np.array([[1] * 4, [1] * 4]),  # GFDL-ESM2M family, RCP4.5
                )
            ),
            "GCM-True": np.concatenate(
                (
                    np.array(
                        [
                            [0.02777, 0.04166, 0.05555, 0.08333],
                            [0.02777, 0.04166, 0.05555, 0.08333],
                            [0.02777, 0.04166, 0.05555, 0.08333],
                            [0.08333, 0.125, 0.16666, 0.25],
                            [0.08333, 0, 0, 0],
                        ]
                    ),  # CanESM2 family, RCP4.5
                    np.array(
                        [
                            [0.00854, 0.00925, 0.00925, 0.02777],
                            [0.00854, 0.00925, 0.00925, 0.02777],
                            [0.00854, 0.00925, 0.00925, 0.02777],
                            [0.02564, 0.02777, 0.02777, 0.08333],
                            [0.02564, 0, 0, 0],
                        ]
                    ),  # CanESM2 family, RCP8.5
                    np.repeat(np.array([[0.02564, 0.02777, 0.02777, 0.08333]]), 10, 0),  # ClimEx
                    np.repeat(np.array([[0.05, 0.05, 0.06666, 0]]), 5, 0),  # CSIRO-Mk6, RCP4.5
                    np.array([[0.33333, 0.33333, 0.33333, 0]]),  # CSIRO2, RCP8.5
                    np.array([[0.16666, 0.16666, 0.16666, 0], [0.16666, 0.16666, 0.16666, 0]]),  # EC-EARTH RCMs, RCP8.5
                    np.array([[0.125, 0.125, 0, 0], [0.125, 0.125, 0, 0]]),  # GFDL-ESM2G family, RCP4.5
                    np.array([[0.125, 0.125, 0.16666, 0.25], [0.125, 0.125, 0.16666, 0.25]]),  # GFDL-ESM2M family, RCP4.5
                )
            ),
            "GCM-False": np.concatenate(
                (
                    np.array(
                        [
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.0625] * 1 + [0.07142] * 3,
                            [0.0625] * 1 + [0] * 3,
                        ]
                    ),  # CanESM2 family, RCP4.5
                    np.array(
                        [
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.0625] * 1 + [0.07142] * 3,
                            [0.0625] * 1 + [0] * 3,
                        ]
                    ),  # CanESM2 family, RCP8.5
                    np.repeat(np.array([[0.0625] * 1 + [0.07142] * 3]), 10, 0),  # ClimEx
                    np.repeat(np.array([[0.2] * 3 + [0] * 1]), 5, 0),  # CSIRO-Mk6, RCP4.5
                    np.array([[1] * 3 + [0] * 1]),  # CSIRO2, RCP8.5
                    np.array([[0.5] * 3 + [0] * 1, [0.5] * 3 + [0] * 1]),  # EC-EARTH RCMs, RCP8.5
                    np.array([[0.5] * 2 + [0] * 2, [0.5] * 2 + [0] * 2]),  # GFDL-ESM2G family, RCP4.5
                    np.array([[0.5] * 4, [0.5] * 4]),  # GFDL-ESM2M family, RCP4.5
                )
            ),
            "institution-True": np.concatenate(
                (
                    np.array(
                        [
                            [0.03703, 0.05555, 0.05555, 0.08333],
                            [0.03703, 0.05555, 0.05555, 0.08333],
                            [0.03703, 0.05555, 0.05555, 0.08333],
                            [0.11111, 0.16666, 0.16666, 0.25],
                            [0.11111, 0, 0, 0],
                        ]
                    ),  # CanESM2 family, RCP4.5
                    np.array(
                        [
                            [0.00854, 0.00925, 0.00925, 0.02777],
                            [0.00854, 0.00925, 0.00925, 0.02777],
                            [0.00854, 0.00925, 0.00925, 0.02777],
                            [0.02564, 0.02777, 0.02777, 0.08333],
                            [0.02564, 0, 0, 0],
                        ]
                    ),  # CanESM2 family, RCP8.5
                    np.repeat(np.array([[0.02564, 0.02777, 0.02777, 0.08333]]), 10, 0),  # ClimEx
                    np.repeat(np.array([[0.06666, 0.06666, 0.06666, 0]]), 5, 0),  # CSIRO-Mk6, RCP4.5
                    np.array([[0.33333, 0.33333, 0.33333, 0]]),  # CSIRO2, RCP8.5
                    np.array([[0.16666, 0.16666, 0.16666, 0], [0.16666, 0.16666, 0.16666, 0]]),  # EC-EARTH RCMs, RCP8.5
                    np.array([[0.08333, 0.08333, 0, 0], [0.08333, 0.08333, 0, 0]]),  # GFDL-ESM2G family, RCP4.5
                    np.array(
                        [
                            [0.08333, 0.08333, 0.16666, 0.25],
                            [0.08333, 0.08333, 0.16666, 0.25],
                        ]
                    ),  # GFDL-ESM2M family, RCP4.5
                )
            ),
            "institution-False": np.concatenate(
                (
                    np.array(
                        [
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.0625] * 1 + [0.07142] * 3,
                            [0.0625] * 1 + [0] * 3,
                        ]
                    ),  # CanESM2 family, RCP4.5
                    np.array(
                        [
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.02083] * 1 + [0.02380] * 3,
                            [0.0625] * 1 + [0.07142] * 3,
                            [0.0625] * 1 + [0] * 3,
                        ]
                    ),  # CanESM2 family, RCP8.5
                    np.repeat(np.array([[0.0625] * 1 + [0.07142] * 3]), 10, 0),  # ClimEx
                    np.repeat(np.array([[0.1] * 3 + [0] * 1]), 5, 0),  # CSIRO-Mk6, RCP4.5
                    np.array([[0.5] * 3 + [0] * 1]),  # CSIRO2, RCP8.5
                    np.array([[0.5] * 3 + [0] * 1, [0.5] * 3 + [0] * 1]),  # EC-EARTH RCMs, RCP8.5
                    np.array([[0.25] * 2 + [0] * 2, [0.25] * 2 + [0] * 2]),  # GFDL-ESM2G family, RCP4.5
                    np.array([[0.25] * 2 + [0.5] * 2, [0.25] * 2 + [0.5] * 2]),  # GFDL-ESM2M family, RCP4.5
                )
            ),
        }

        answer = answers[f"{independence_level}-{exp_weights}"]
        if skipna:
            answer = answer[:, 0]

        return answer

    @pytest.mark.parametrize(
        "independence_level, exp_weights, skipna",
        [
            ("model", True, True),
            ("model", False, True),
            ("GCM", True, True),
            ("GCM", False, True),
            ("institution", True, True),
            ("institution", False, True),
            ("model", True, False),
            ("model", False, False),
            ("GCM", True, False),
            ("GCM", False, False),
            ("institution", True, False),
            ("institution", False, False),
        ],
    )
    def test_generate_weights(self, independence_level, exp_weights, skipna):
        out = xs.generate_weights(
            self.ens,
            independence_level=independence_level,
            balance_experiments=exp_weights,
            skipna=skipna,
        )

        answer = self.make_answer(independence_level, exp_weights, skipna)
        np.testing.assert_array_almost_equal(out, answer, decimal=4)

    def test_changing_horizon(self):
        ens = deepcopy(self.ens)
        ds = ens["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"]
        hz = xr.DataArray(["1981-2010", "2041-2070", "+2C", "not_the_same"], dims="horizon")
        ens["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"] = ds.assign_coords({"horizon": hz})

        out = xs.generate_weights(ens, skipna=False)
        assert len(out.horizon) == 5
        # Should all be 0s, except for CCCma-CanESM2-rcp45-r1i1p1-CanESM2
        np.testing.assert_array_equal(
            out.sel(realization="CCCma-CanESM2-rcp45-r1i1p1-CanESM2", horizon="not_the_same"),
            1,
        )
        assert all(
            out.sel(
                realization=~out.realization.isin(["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"]),
                horizon="not_the_same",
            )
            == 0
        )

    @pytest.mark.parametrize(
        "standardize, skipna, attribute_weights",
        [
            (True, True, None),
            (True, False, None),
            (False, True, None),
            (
                True,
                True,
                {"institution": {"CCCma": 2, "ECMWF": 0, "GFDL": 5}},
            ),
        ],
    )
    def test_standardize(self, standardize, skipna, attribute_weights):
        if attribute_weights:
            datasets = self.ens_rcm
        else:
            datasets = self.ens
        out = xs.generate_weights(
            datasets,
            standardize=standardize,
            skipna=skipna,
            attribute_weights=attribute_weights,
        )
        if standardize:
            np.testing.assert_allclose(out.sum(), 1 if skipna else 4)

        else:
            np.testing.assert_allclose(out.sum(), 10)

    def test_spatial(self, datablock_3d):
        a = datablock_3d(np.random.rand(10, 10, 10), "tas", "lon", 0, "lat", 0, as_dataset=True)
        a.attrs = {
            "cat:institution": "CCCma",
            "cat:source": "CanESM5",
            "cat:experiment": "ssp585",
            "cat:member": "r1i1p1f1",
        }
        b = datablock_3d(np.random.rand(10, 10, 10), "tas", "lon", 0, "lat", 0, as_dataset=True)
        b.attrs = {
            "cat:institution": "CCCma",
            "cat:source": "CanESM5",
            "cat:experiment": "ssp585",
            "cat:member": "r2i1p1f1",
        }
        c = datablock_3d(np.random.rand(10, 10, 10), "tas", "lon", 0, "lat", 0, as_dataset=True)
        c.attrs = {
            "cat:institution": "CSIRO",
            "cat:source": "ACCESS-ESM1-5",
            "cat:experiment": "ssp585",
            "cat:member": "r1i1p1f1",
        }
        ens = [a, b, c]

        np.testing.assert_allclose(xs.generate_weights(ens, skipna=True), [0.5, 0.5, 1.0])
        with pytest.warns(UserWarning, match="Dataset 0 has dimensions that are not "):
            np.testing.assert_allclose(
                xs.generate_weights(ens, skipna=False),
                [[0.5] * 10, [0.5] * 10, [1.0] * 10],
            )

    def test_errors(self):
        # Bad input
        with pytest.raises(ValueError, match="'independence_level' should be between"):
            xs.generate_weights(self.ens, independence_level="foo")

        ens2 = deepcopy(self.ens)
        ens2["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"] = ens2["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"].expand_dims("time")
        with pytest.raises(
            ValueError,
            match="Expected either 'time' or 'horizon' as an extra dimension",
        ):
            xs.generate_weights(ens2, skipna=False)
        xs.generate_weights(ens2, skipna=True)  # Should work

        # Required attributes
        ens2 = deepcopy(self.ens)
        ens2["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"].attrs["cat:source"] = None
        with pytest.raises(
            ValueError,
            match="The 'cat:source' or 'cat:driving_model' attribute is missing",
        ):
            xs.generate_weights(ens2)
        ens2 = deepcopy(self.ens)
        ens2["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"].attrs["cat:experiment"] = None
        with pytest.raises(ValueError, match="The 'cat:experiment' attribute is missing"):
            xs.generate_weights(ens2, balance_experiments=True)
        ens2 = deepcopy(self.ens)
        ens2["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"].attrs["cat:institution"] = None
        with pytest.raises(ValueError, match="The 'cat:institution' attribute is missing"):
            xs.generate_weights(ens2, independence_level="institution")

        # Optional, but recommended attributes
        ens2 = deepcopy(self.ens)
        for e in ens2:
            ens2[e].attrs["cat:experiment"] = None
        with pytest.warns(
            UserWarning,
            match="The 'cat:experiment' attribute is missing from all datasets",
        ):
            xs.generate_weights(ens2, balance_experiments=False)
        ens2 = deepcopy(self.ens)
        ens2["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"].attrs["cat:member"] = None
        with pytest.warns(
            UserWarning,
            match="The 'cat:member' attribute is inconsistent across datasets.",
        ):
            xs.generate_weights(ens2)

    @staticmethod
    def answer_attribute_weight(can45, can85, clim, csi45, csi85, ec85, gfd45):
        ans = np.concatenate(
            (
                np.array(
                    np.split(
                        np.array(
                            [
                                0.08333,
                                0.25,
                                0.25,
                                0.25,
                                0.25,
                            ]
                        ),
                        5,
                    )
                )
                * can45,  # CanESM2 family, RCP4.5
                np.array(
                    np.split(
                        np.array(
                            [
                                0.08333,
                                0.25,
                                0.25,
                                0.25,
                                0.25,
                            ]
                        ),
                        5,
                    )
                )
                * can85,  # CanESM2 family, RCP8.5
                np.array(np.split(np.repeat(np.array(0.08333), 10, 0), 10)) * clim,  # ClimEx
                np.array(np.split(np.repeat(0.2, 5, 0), 5)) * csi45,  # CSIRO-Mk6, RCP4.5
                np.array([1], ndmin=2) * csi85,  # CSIRO2, RCP8.5
                np.array(np.split(np.array([1, 1]), 2)) * ec85,  # EC-EARTH RCMs, RCP8.5
                np.array(np.split(np.array([0.5, 0.5]), 2)) * gfd45,  # GFDL-ESM2G family, RCP4.5
                np.array(np.split(np.array([1, 1]), 2)) * gfd45,  # GFDL-ESM2M family, RCP4.5
            )
        )
        return np.squeeze(ans)

    @pytest.mark.parametrize(
        "coefs, weights",
        [
            (
                {
                    "can45": 2,
                    "can85": 2,
                    "clim": 2,
                    "csi45": 3,
                    "csi85": 3,
                    "ec85": 0,
                    "gfd45": 5,
                },
                {"institution": {"CCCma": 2, "CSIRO-QCCCE": 3, "ECMWF": 0, "GFDL": 5}},
            ),
            (
                {
                    "can45": 2,
                    "can85": 2,
                    "clim": 2,
                    "csi45": 1,
                    "csi85": 1,
                    "ec85": 1,
                    "gfd45": 1,
                },
                {"institution": {"CCCma": 2, "others": 1}},
            ),
            (
                {
                    "can45": 2,
                    "can85": 1,
                    "clim": 1,
                    "csi45": 2,
                    "csi85": 1,
                    "ec85": 1,
                    "gfd45": 2,
                },
                {"experiment": {"rcp45": 2, "rcp85": 1}},
            ),
            (
                {
                    "can45": [0, 0, 1, 1],
                    "can85": [0, 1, 0, 1],
                    "clim": [0, 1, 0, 1],
                    "csi45": [0, 0, 1, 1],
                    "csi85": [0, 1, 0, 1],
                    "ec85": [0, 1, 0, 1],
                    "gfd45": [0, 0, 1, 1],
                },
                {
                    "experiment": xr.DataArray(
                        data=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                        dims=["horizon", "experiment"],
                        coords=dict(
                            horizon=["1981-2010", "2041-2070", "+2C", "+4C"],
                            experiment=["rcp45", "rcp85"],
                        ),
                        attrs=dict(
                            description="Experiment weight through horizons",
                            units="",
                        ),
                    )
                },
            ),
            (
                {
                    "can45": [0, 0, 1, 1],
                    "can85": [0, 1, 0, 1],
                    "clim": [0, 1, 0, 1],
                    "csi45": [0, 0, 1, 1],
                    "csi85": [0, 1, 0, 1],
                    "ec85": [0, 1, 0, 1],
                    "gfd45": [0, 0, 1, 1],
                },
                {
                    "experiment": xr.DataArray(
                        data=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                        dims=["horizon", "experiment"],
                        coords=dict(
                            horizon=["1981-2010", "2041-2070", "+2C", "+4C"],
                            experiment=["rcp45", "others"],
                        ),
                        attrs=dict(
                            description="Experiment weight through horizons",
                            units="",
                        ),
                    )
                },
            ),
        ],
    )
    def test_attribute_weight(self, coefs, weights):
        # only test for RCMs

        # generate ans
        ans = self.answer_attribute_weight(**coefs)
        ind = [index for index, value in enumerate(list(self.ens.keys())) if value in list(self.ens_rcm.keys())]
        ans_rcms = ans[ind]
        if len(ans_rcms.shape) > 1:
            ans_rcms = ans_rcms.transpose()
        out = xs.generate_weights(self.ens_rcm, attribute_weights=weights)
        np.testing.assert_array_almost_equal(out, ans_rcms, decimal=4)

    def test_attribute_weight_error(self):
        # Required attributes
        with pytest.raises(
            ValueError,
            match="Attribute_weights should be dict or xr.DataArray.",
        ):
            xs.generate_weights(self.ens_rcm, attribute_weights={"experiment": [1, 2, 3]})
        with pytest.raises(
            ValueError,
            match="The test attribute is missing from some simulations.",
        ):
            xs.generate_weights(self.ens_rcm, attribute_weights={"test": {"CCCma": 2, "others": 1}})
        with pytest.raises(
            ValueError,
            match="The institution ECMWF or others are not in the attribute_weights dict.",
        ):
            xs.generate_weights(self.ens_rcm, attribute_weights={"institution": {"CCCma": 2, "GFDL": 1}})
        with pytest.raises(ValueError, match="experiment is not in the xr.DataArray coords."):
            xs.generate_weights(
                self.ens_rcm,
                attribute_weights={
                    "experiment": xr.DataArray(
                        data=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                        dims=["horizon", "test"],
                        coords=dict(
                            horizon=["1981-2010", "2041-2070", "+2C", "+4C"],
                            test=["rcp45", "rcp85"],
                        ),
                    )
                },
            )
        with pytest.raises(
            ValueError,
            match="The experiment DataArray has more than one coord dimension to apply weights",
        ):
            xs.generate_weights(
                self.ens_rcm,
                attribute_weights={
                    "experiment": xr.DataArray(
                        data=np.array(
                            [
                                [[0, 0], [0, 1]],
                                [[0, 1], [2, 1]],
                                [[1, 0], [1, 2]],
                                [[1, 1], [2, 2]],
                            ]
                        ),
                        dims=["horizon", "experiment", "test"],
                        coords=dict(
                            horizon=["1981-2010", "2041-2070", "+2C", "+4C"],
                            experiment=["rcp45", "rcp85"],
                            test=[0, 1],
                        ),
                    )
                },
            )
        with pytest.raises(
            ValueError,
            match="The experiment rcp85 or others are not in the attribute_weights datarray coords.",
        ):
            xs.generate_weights(
                self.ens_rcm,
                attribute_weights={
                    "experiment": xr.DataArray(
                        data=np.array([[0], [0], [1], [1]]),
                        dims=["horizon", "experiment"],
                        coords=dict(
                            horizon=["1981-2010", "2041-2070", "+2C", "+4C"],
                            experiment=["rcp45"],
                        ),
                    )
                },
            )
            # mismatch RCMs & GCMs
            with pytest.raises(
                NotImplementedError,
                match="Management of RCM and GCM in same datasets dictionary not yet implemented with attribute_weights.",
            ):
                xs.generate_weights(
                    self.ens,
                    attribute_weights={"institution": {"CCCma": 2, "others": 1}},
                )
            # warning mismatch attribute_weights & independance_level
            with pytest.warns(
                UserWarning,
                match="The institution weights do not match the model independance_level",
            ):
                xs.generate_weights(
                    self.ens_rcm,
                    attribute_weights={"institution": {"CCCma": 2, "others": 1}},
                )
            with pytest.warns(
                UserWarning,
                match="Key experiment given in attribute_weights without argument balance_experiments=True",
            ):
                xs.generate_weights(
                    self.ens_rcm,
                    attribute_weights={"experiment": {"rcp45": 2, "rcp85": 1}},
                )


class TestEnsemblePartition:
    @pytest.mark.skipif(xe is None, reason="xesmf needed for testing regrdding")
    def test_build_partition_data(self, samplecat, tmp_path):
        # test subset
        datasets = samplecat.search(variable="tas", member="r1i1p1f1").to_dataset_dict(xarray_open_kwargs={"engine": "h5netcdf"})
        ds = xs.ensembles.build_partition_data(
            datasets=datasets,
            partition_dim=["source", "experiment"],
            subset_kw=dict(name="mtl", method="gridpoint", lat=[45.0], lon=[-74]),
            rename_dict={"source": "new-name"},
        )

        assert ds.sizes == {"time": 730, "scenario": 4, "new-name": 1}
        assert ds.attrs["cat:processing_level"] == "partition-ensemble"

        # test regrid
        ds_grid = xe.util.cf_grid_2d(-75, -74, 0.25, 45, 48, 0.55)
        datasets = samplecat.search(variable="tas", member="r1i1p1f1").to_dataset_dict(xarray_open_kwargs={"engine": "h5netcdf"})
        ds = xs.ensembles.build_partition_data(
            datasets=datasets,
            regrid_kw=dict(ds_grid=ds_grid, weights_location=tmp_path),
            to_level="test",
        )

        assert ds.sizes == {
            "scenario": 4,
            "model": 1,
            "time": 730,
            "lat": 5,
            "lon": 4,
        }
        assert [i for i in ds.data_vars] == ["tas"]
        assert ds.attrs["cat:processing_level"] == "test"

    def test_partition_from_catalog(self, samplecat):
        datasets = samplecat.search(variable="tas", member="r1i1p1f1")
        ds_from_dict = xs.ensembles.build_partition_data(
            datasets=datasets.to_dataset_dict(xarray_open_kwargs={"engine": "h5netcdf"}),
            partition_dim=["source", "experiment"],
            subset_kw=dict(name="mtl", method="gridpoint", lat=[45.0], lon=[-74]),
        )

        ds_from_cat = xs.ensembles.build_partition_data(
            datasets=datasets,
            partition_dim=["source", "experiment"],
            subset_kw=dict(name="mtl", method="gridpoint", lat=[45.0], lon=[-74]),
            to_dataset_kw=dict(xarray_open_kwargs={"engine": "h5netcdf"}),
        )
        # fix order
        ds_from_cat = ds_from_cat[["time", "model", "scenario", "tas"]]
        ds_from_cat["tas"] = ds_from_cat["tas"].transpose("scenario", "model", "time")

        assert ds_from_dict.equals(ds_from_cat)

    def test_realization_partition(self, samplecat):
        datasets = samplecat.search(variable="tas").to_dataset_dict(xarray_open_kwargs={"engine": "h5netcdf"})
        ds = xs.ensembles.build_partition_data(
            datasets=datasets,
            partition_dim=["realization", "experiment"],
            subset_kw=dict(name="mtl", method="gridpoint", lat=[45.0], lon=[-74]),
        )

        assert "NCC_NorESM2-MM_r1i1p1f1" in ds.model.values
        assert ds.sizes == {"time": 730, "scenario": 4, "model": 2}


class TestReduceEnsemble:
    def test_with_criteria(self):
        ds = xr.open_dataset(nimbus().fetch("EnsembleReduce/TestEnsReduceCriteria.nc"))
        selected, clusters, fig_data = xs.reduce_ensemble(ds["data"], method="kmeans", max_clusters=3)
        assert selected.shape == (3,)
        np.testing.assert_array_equal(selected, [4, 7, 23])
        assert len(clusters) == 3
        assert fig_data is not None

    @pytest.mark.parametrize("horizon", ["1981-2010", "2021-2050"])
    def test_without_criteria(self, horizon):
        datasets = {
            "ACCESS": "EnsembleStats/BCCAQv2+ANUSPLIN300_ACCESS1-0_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
            "BNU-ESM": "EnsembleStats/BCCAQv2+ANUSPLIN300_BNU-ESM_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
            "CCSM4-r1": "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r1i1p1_1950-2100_tg_mean_YS.nc",
            "CCSM4-r2": "EnsembleStats/BCCAQv2+ANUSPLIN300_CCSM4_historical+rcp45_r2i1p1_1950-2100_tg_mean_YS.nc",
            "CNRM-CM5": "EnsembleStats/BCCAQv2+ANUSPLIN300_CNRM-CM5_historical+rcp45_r1i1p1_1970-2050_tg_mean_YS.nc",
        }
        for d in datasets:
            ds = xr.open_dataset(nimbus().fetch(datasets[d])).isel(lon=slice(0, 4), lat=slice(0, 4))
            ds = xs.climatological_op(
                ds,
                op="mean",
                window=30,
                periods=[["1981", "2010"], ["2021", "2050"]],
                horizons_as_dim=True,
            ).drop_vars("time")
            datasets[d] = xs.compute_deltas(ds, reference_horizon="1981-2010")

        selected, clusters, fig_data = xs.reduce_ensemble(datasets, method="kkz", horizons=[horizon], num_select=4)
        assert selected.shape == (4,)
        answer = ["ACCESS", "BNU-ESM", "CNRM-CM5", "CCSM4-r1"] if horizon == "2021-2050" else ["ACCESS", "BNU-ESM", "CCSM4-r1", "CCSM4-r2"]
        np.testing.assert_array_equal(selected, answer)
        assert clusters == {}
        assert fig_data == {}

    def test_errors(self):
        ds = xr.open_dataset(nimbus().fetch("EnsembleReduce/TestEnsReduceCriteria.nc"))
        with pytest.raises(ValueError, match="Data must have a 'horizon' dimension to be subsetted."):
            xs.reduce_ensemble(ds["data"], method="kmeans", horizons=["1981-2010"], max_clusters=3)
