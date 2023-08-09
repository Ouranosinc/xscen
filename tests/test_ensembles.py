from copy import deepcopy

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
                            f"{institution}-{gcm}-{ds.attrs['cat:experiment']}-{member}-{model}"
                        ] = ds

                        if gcm == "CanESM2":
                            ds2 = ds.copy()
                            ds2.attrs["cat:experiment"] = "rcp85"
                            out[
                                f"{institution}-{gcm}-{ds2.attrs['cat:experiment']}-{member}-{model}"
                            ] = ds2

                            if model == "CRCM5" and member == "r1i1p1":
                                for i in range(10):
                                    ds2 = ds.copy()
                                    ds2.attrs["cat:experiment"] = "rcp85"
                                    ds2.attrs["cat:member"] = f"xr{i}r1i1p1"
                                    ds2.attrs["cat:activity"] = "ClimEx"
                                    out[
                                        f"{institution}-{gcm}-{ds2.attrs['cat:experiment']}-{ds2.attrs['cat:member']}-{model}"
                                    ] = ds2
                        elif gcm == "CSIRO-Mk3-6-0" and member == "r1i1p1":
                            ds2 = ds.copy()
                            ds2.attrs["cat:experiment"] = "rcp85"
                            ds2.attrs["cat:source"] = "CSIRO2"
                            out[
                                f"{institution}-CSIRO2-{member}-{ds2.attrs['cat:experiment']}-CSIRO2"
                            ] = ds2
        return {o: out[o] for o in sorted(out)}

    ens = make_ensemble.__func__()

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
                    np.repeat(
                        np.array([[0.01515, 0.01515, 0.01515, 0.03030]]), 10, 0
                    ),  # ClimEx
                    np.repeat(
                        np.array([[0.02857, 0.02857, 0.03333, 0]]), 5, 0
                    ),  # CSIRO-Mk6, RCP4.5
                    np.array([[0.16666, 0.16666, 0.16666, 0]]),  # CSIRO2, RCP8.5
                    np.array(
                        [[0.16666, 0.16666, 0.16666, 0], [0.16666, 0.16666, 0.16666, 0]]
                    ),  # EC-EARTH RCMs, RCP8.5
                    np.array(
                        [[0.07142, 0.07142, 0, 0], [0.07142, 0.07142, 0, 0]]
                    ),  # GFDL-ESM2G family, RCP4.5
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
                    np.repeat(
                        np.array([[0.2] * 3 + [0] * 1]), 5, 0
                    ),  # CSIRO-Mk6, RCP4.5
                    np.array([[1] * 3 + [0] * 1]),  # CSIRO2, RCP8.5
                    np.array(
                        [[1] * 3 + [0] * 1, [1] * 3 + [0] * 1]
                    ),  # EC-EARTH RCMs, RCP8.5
                    np.array(
                        [[0.5] * 2 + [0] * 2, [0.5] * 2 + [0] * 2]
                    ),  # GFDL-ESM2G family, RCP4.5
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
                    np.repeat(
                        np.array([[0.02564, 0.02777, 0.02777, 0.08333]]), 10, 0
                    ),  # ClimEx
                    np.repeat(
                        np.array([[0.05, 0.05, 0.06666, 0]]), 5, 0
                    ),  # CSIRO-Mk6, RCP4.5
                    np.array([[0.33333, 0.33333, 0.33333, 0]]),  # CSIRO2, RCP8.5
                    np.array(
                        [[0.16666, 0.16666, 0.16666, 0], [0.16666, 0.16666, 0.16666, 0]]
                    ),  # EC-EARTH RCMs, RCP8.5
                    np.array(
                        [[0.125, 0.125, 0, 0], [0.125, 0.125, 0, 0]]
                    ),  # GFDL-ESM2G family, RCP4.5
                    np.array(
                        [[0.125, 0.125, 0.16666, 0.25], [0.125, 0.125, 0.16666, 0.25]]
                    ),  # GFDL-ESM2M family, RCP4.5
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
                    np.repeat(
                        np.array([[0.0625] * 1 + [0.07142] * 3]), 10, 0
                    ),  # ClimEx
                    np.repeat(
                        np.array([[0.2] * 3 + [0] * 1]), 5, 0
                    ),  # CSIRO-Mk6, RCP4.5
                    np.array([[1] * 3 + [0] * 1]),  # CSIRO2, RCP8.5
                    np.array(
                        [[0.5] * 3 + [0] * 1, [0.5] * 3 + [0] * 1]
                    ),  # EC-EARTH RCMs, RCP8.5
                    np.array(
                        [[0.5] * 2 + [0] * 2, [0.5] * 2 + [0] * 2]
                    ),  # GFDL-ESM2G family, RCP4.5
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
                    np.repeat(
                        np.array([[0.02564, 0.02777, 0.02777, 0.08333]]), 10, 0
                    ),  # ClimEx
                    np.repeat(
                        np.array([[0.06666, 0.06666, 0.06666, 0]]), 5, 0
                    ),  # CSIRO-Mk6, RCP4.5
                    np.array([[0.33333, 0.33333, 0.33333, 0]]),  # CSIRO2, RCP8.5
                    np.array(
                        [[0.16666, 0.16666, 0.16666, 0], [0.16666, 0.16666, 0.16666, 0]]
                    ),  # EC-EARTH RCMs, RCP8.5
                    np.array(
                        [[0.08333, 0.08333, 0, 0], [0.08333, 0.08333, 0, 0]]
                    ),  # GFDL-ESM2G family, RCP4.5
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
                    np.repeat(
                        np.array([[0.0625] * 1 + [0.07142] * 3]), 10, 0
                    ),  # ClimEx
                    np.repeat(
                        np.array([[0.1] * 3 + [0] * 1]), 5, 0
                    ),  # CSIRO-Mk6, RCP4.5
                    np.array([[0.5] * 3 + [0] * 1]),  # CSIRO2, RCP8.5
                    np.array(
                        [[0.5] * 3 + [0] * 1, [0.5] * 3 + [0] * 1]
                    ),  # EC-EARTH RCMs, RCP8.5
                    np.array(
                        [[0.25] * 2 + [0] * 2, [0.25] * 2 + [0] * 2]
                    ),  # GFDL-ESM2G family, RCP4.5
                    np.array(
                        [[0.25] * 2 + [0.5] * 2, [0.25] * 2 + [0.5] * 2]
                    ),  # GFDL-ESM2M family, RCP4.5
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
            experiment_weights=exp_weights,
            skipna=skipna,
        )

        answer = self.make_answer(independence_level, exp_weights, skipna)
        np.testing.assert_array_almost_equal(out, answer, decimal=4)

    def test_changing_horizon(self):
        ens = deepcopy(self.ens)
        ds = ens["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"]
        hz = xr.DataArray(
            ["1981-2010", "2041-2070", "+2C", "not_the_same"], dims="horizon"
        )
        ens["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"] = ds.assign_coords({"horizon": hz})

        out = xs.generate_weights(ens, skipna=False)
        assert len(out.horizon) == 5
        # Should all be 0s, except for CCCma-CanESM2-rcp45-r1i1p1-CanESM2
        np.testing.assert_array_equal(
            out.sel(
                realization="CCCma-CanESM2-rcp45-r1i1p1-CanESM2", horizon="not_the_same"
            ),
            1,
        )
        assert all(
            out.sel(
                realization=~out.realization.isin(
                    ["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"]
                ),
                horizon="not_the_same",
            )
            == 0
        )

    @pytest.mark.parametrize(
        "standardize, skipna", [(True, True), (True, False), (False, True)]
    )
    def test_standardize(self, standardize, skipna):
        out = xs.generate_weights(self.ens, standardize=standardize, skipna=skipna)
        if standardize:
            np.testing.assert_allclose(out.sum(), 1 if skipna else 4)
        else:
            np.testing.assert_allclose(out.sum(), 10)

    def test_spatial(self, datablock_3d):
        a = datablock_3d(
            np.random.rand(10, 10, 10), "tas", "lon", 0, "lat", 0, as_dataset=True
        )
        a.attrs = {
            "cat:institution": "CCCma",
            "cat:source": "CanESM5",
            "cat:experiment": "ssp585",
            "cat:member": "r1i1p1f1",
        }
        b = datablock_3d(
            np.random.rand(10, 10, 10), "tas", "lon", 0, "lat", 0, as_dataset=True
        )
        b.attrs = {
            "cat:institution": "CCCma",
            "cat:source": "CanESM5",
            "cat:experiment": "ssp585",
            "cat:member": "r2i1p1f1",
        }
        c = datablock_3d(
            np.random.rand(10, 10, 10), "tas", "lon", 0, "lat", 0, as_dataset=True
        )
        c.attrs = {
            "cat:institution": "CSIRO",
            "cat:source": "ACCESS-ESM1-5",
            "cat:experiment": "ssp585",
            "cat:member": "r1i1p1f1",
        }
        ens = [a, b, c]

        np.testing.assert_allclose(
            xs.generate_weights(ens, skipna=True), [0.5, 0.5, 1.0]
        )
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
        ens2["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"] = ens2[
            "CCCma-CanESM2-rcp45-r1i1p1-CanESM2"
        ].expand_dims("time")
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
        with pytest.raises(
            ValueError, match="The 'cat:experiment' attribute is missing"
        ):
            xs.generate_weights(ens2, experiment_weights=True)
        ens2 = deepcopy(self.ens)
        ens2["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"].attrs["cat:institution"] = None
        with pytest.raises(
            ValueError, match="The 'cat:institution' attribute is missing"
        ):
            xs.generate_weights(ens2, independence_level="institution")

        # Optional, but recommended attributes
        ens2 = deepcopy(self.ens)
        for e in ens2:
            ens2[e].attrs["cat:experiment"] = None
        with pytest.warns(
            UserWarning,
            match="The 'cat:experiment' attribute is missing from all datasets",
        ):
            xs.generate_weights(ens2, experiment_weights=False)
        ens2 = deepcopy(self.ens)
        ens2["CCCma-CanESM2-rcp45-r1i1p1-CanESM2"].attrs["cat:member"] = None
        with pytest.warns(
            UserWarning,
            match="The 'cat:member' attribute is inconsistent across datasets.",
        ):
            xs.generate_weights(ens2)
