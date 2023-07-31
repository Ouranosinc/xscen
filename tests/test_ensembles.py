import numpy as np
import pytest
import xarray as xr
from xclim.testing.helpers import test_timeseries as timeseries

import xscen as xs


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

    @pytest.mark.parametrize(
        "weights, to_level", [(None, None), (np.arange(1, 11), "for_tests")]
    )
    def test_weights(self, weights, to_level):
        ens = self.make_ensemble(10)
        if weights is not None:
            weights = xr.DataArray(
                weights, dims="realization", coords={"realization": np.arange(0, 10)}
            )

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
        assert (
            out.attrs["cat:processing_level"] == to_level
            if to_level is not None
            else "ensemble"
        )
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

    def test_change_significance(self):
        # TODO: Possible nearby changes to xclim.ensembles.change_significance would break this test.
        pass

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
            "CSIRO-Mk3-6-0": 10,
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
                    has_gcm = (
                        []
                        if (
                            (gcm == "CanESM2" and member == "r3i1p1")
                            or gcm == "EC-EARTH"
                        )
                        else [gcm]
                    )
                    rcm = [r for r in rcms if f"{gcm}-{member}" in rcms[r]]
                    for model in has_gcm + rcm:
                        ds = timeseries(
                            np.arange(1, 5),
                            variable="tas",
                            start="1981-01-01",
                            freq="30YS",
                            as_dataset=True,
                        )
                        ds["horizon"] = xr.DataArray(
                            ["1981-2010", "2041-2070", "+2C", "+4C"], dims="time"
                        )
                        ds = ds.swap_dims({"time": "horizon"}).drop("time")
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

                        out[
                            f"{ds.attrs['cat:activity']}-{ds.attrs['cat:experiment']}-{institution}-{gcm}-{model}-{member}"
                        ] = ds

                        if gcm == "CanESM2":
                            ds2 = ds.copy()
                            ds2.attrs["cat:experiment"] = "rcp85"
                            out[
                                f"{ds2.attrs['cat:activity']}-{ds2.attrs['cat:experiment']}-{institution}-{gcm}-{model}-{member}"
                            ] = ds2

                            if model == "CRCM5" and member == "r1i1p1":
                                for i in range(50):
                                    ds2 = ds.copy()
                                    ds2.attrs["cat:experiment"] = "rcp85"
                                    ds2.attrs["cat:member"] = f"r{i}-r1i1p1"
                                    ds2.attrs["cat:activity"] = "ClimEx"
                                    out[
                                        f"{ds2.attrs['cat:activity']}-{ds2.attrs['cat:experiment']}-{institution}-{gcm}-{model}-{ds2.attrs['cat:member']}"
                                    ] = ds2
                        elif gcm == "CSIRO-Mk3-6-0" and member == "r1i1p1":
                            ds2 = ds.copy()
                            ds2.attrs["cat:experiment"] = "rcp85"
                            ds2.attrs["cat:source"] = "CSIRO2"
                            out[
                                f"{ds2.attrs['cat:activity']}-{ds2.attrs['cat:experiment']}-{institution}-CSIRO2-CSIRO2-{member}"
                            ] = ds2
        return out

    @pytest.mark.parametrize(
        "independence_level, split_exp",
        [
            ("all", True),
            ("all", False),
            ("GCM", True),
            ("GCM", False),
            ("institution", True),
            ("institution", False),
        ],
    )
    def test_generate_weights(self, independence_level, split_exp):
        ens = self.make_ensemble()
        out = xs.generate_weights(
            ens, independence_level=independence_level, split_experiments=split_exp
        )
        out = out.sortby("realization")

        # Could be more concise, but this is much easier to understand
        answers = {
            "CanESM2-rcp45-x2": {
                "all-True": [1 / 2] * 2,
                "all-False": [1 / 4] * 2,
                "GCM-True": [1 / 3 / 3] + [1 / 3],
                "GCM-False": [1 / 3 / 56] + [1 / 56],
                "institution-True": [1 / 3 / 3] + [1 / 3],
                "institution-False": [1 / 3 / 56] + [1 / 56],
            },
            "CSIRO-Mk3-rcp45-x10": {
                "all-True": [1 / 10] * 10,
                "all-False": [1 / 10] * 10,
                "GCM-True": [1 / 10] * 10,
                "GCM-False": [1 / 10] * 10,
                "institution-True": [1 / 10] * 10,
                "institution-False": [1 / 10 / 2] * 10,
            },
            "GFDL-ESM2G-x2": {
                "all-True": [1 / 2] * 2,
                "all-False": [1 / 2] * 2,
                "GCM-True": [1 / 2] * 2,
                "GCM-False": [1 / 2] * 2,
                "institution-True": [1 / 2 / 2] * 2,
                "institution-False": [1 / 2 / 2] * 2,
            },
            "GFDL-ESM2M-x1": {
                "all-True": [1],
                "all-False": [1],
                "GCM-True": [1 / 2],
                "GCM-False": [1 / 2],
                "institution-True": [1 / 2 / 2],
                "institution-False": [1 / 2 / 2],
            },
            "CanESM2-rcp85-x2": {
                "all-True": [1 / 2] * 2,
                "all-False": [1 / 4] * 2,
                "GCM-True": [1 / 3 / 53] + [1 / 53],
                "GCM-False": [1 / 3 / 56] + [1 / 56],
                "institution-True": [1 / 3 / 53] + [1 / 53],
                "institution-False": [1 / 3 / 56] + [1 / 56],
            },
            "CSIRO2-rcp85-x1": {
                "all-True": [1],
                "all-False": [1],
                "GCM-True": [1],
                "GCM-False": [1],
                "institution-True": [1],
                "institution-False": [1 / 2],
            },
            "CanESM2-CRCM5-rcp45-x1": {
                "all-True": [1],
                "all-False": [1 / 52],
                "GCM-True": [1 / 3 / 3],
                "GCM-False": [1 / 3 / 56],
                "institution-True": [1 / 3 / 3],
                "institution-False": [1 / 3 / 56],
            },
            "CanESM2-CanRCM4-rcp45-x2": {
                "all-True": [1 / 2] * 2,
                "all-False": [1 / 4] * 2,
                "GCM-True": [1 / 3 / 3] + [1 / 3],
                "GCM-False": [1 / 3 / 56] + [1 / 56],
                "institution-True": [1 / 3 / 3] + [1 / 3],
                "institution-False": [1 / 3 / 56] + [1 / 56],
            },
            "GFDL-ESM2M-HIRHAM5-x1": {
                "all-True": [1],
                "all-False": [1],
                "GCM-True": [1 / 2],
                "GCM-False": [1 / 2],
                "institution-True": [1 / 2 / 2],
                "institution-False": [1 / 2 / 2],
            },
            "CanESM2-CRCM5-rcp85-x1": {
                "all-True": [1 / 51],
                "all-False": [1 / 52],
                "GCM-True": [1 / 3 / 53],
                "GCM-False": [1 / 3 / 56],
                "institution-True": [1 / 3 / 53],
                "institution-False": [1 / 3 / 56],
            },
            "CanESM2-CanRCM4-rcp85-x2": {
                "all-True": [1 / 2] * 2,
                "all-False": [1 / 4] * 2,
                "GCM-True": [1 / 3 / 53] + [1 / 53],
                "GCM-False": [1 / 3 / 56] + [1 / 56],
                "institution-True": [1 / 3 / 53] + [1 / 53],
                "institution-False": [1 / 3 / 56] + [1 / 56],
            },
            "EC-EARTH-HIRHAM5-x1": {
                "all-True": [1],
                "all-False": [1],
                "GCM-True": [1 / 2],
                "GCM-False": [1 / 2],
                "institution-True": [1 / 2],
                "institution-False": [1 / 2],
            },
            "EC-EARTH-RegCM4-x1": {
                "all-True": [1],
                "all-False": [1],
                "GCM-True": [1 / 2],
                "GCM-False": [1 / 2],
                "institution-True": [1 / 2],
                "institution-False": [1 / 2],
            },
            "ClimEx-CanESM2-CRCM5-rcp85-x50": {
                "all-True": [1 / 51] * 50,
                "all-False": [1 / 52] * 50,
                "GCM-True": [1 / 53] * 50,
                "GCM-False": [1 / 56] * 50,
                "institution-True": [1 / 53] * 50,
                "institution-False": [1 / 56] * 50,
            },
        }

        answer = np.concatenate(
            [answers[k][f"{independence_level}-{split_exp}"] for k in answers]
        )
        np.testing.assert_array_almost_equal(out, answer, decimal=5)
