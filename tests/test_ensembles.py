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
                    "hoizon": xr.DataArray(
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

        if to_level is None:
            out = xs.ensemble_stats(
                ens,
                statistics={
                    "ensemble_mean_std_max_min": None,
                    "ensemble_percentiles": {"split": False},
                },
                weights=weights,
                common_attrs_only=False,
            )
        else:
            out = xs.ensemble_stats(
                ens,
                statistics={
                    "ensemble_mean_std_max_min": None,
                    "ensemble_percentiles": {"split": False},
                },
                weights=weights,
                common_attrs_only=False,
                to_level=to_level,
            )

        assert out.attrs["ensemble_size"] == 10
        assert (
            out.attrs["cat:processing_level"] == to_level
            if to_level is not None
            else "ensemble"
        )
        assert "realization" not in out.dims
        np.testing.assert_array_equal(out.tg_mean_mean, [1.45, 2.45, 3.45, 4.45])
        np.testing.assert_array_almost_equal(out.tg_mean_stdev, [0.2872] * 4, decimal=4)
        np.testing.assert_array_almost_equal(
            out.tg_mean.sel(percentiles=90), [1.81, 2.81, 3.81, 4.81], decimal=2
        )
