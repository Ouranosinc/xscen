import numpy as np

from xscen import aggregate


class TestClimatologicalMean:
    def test_monthly_all_default(self, tas_series_1d):
        ds = tas_series_1d(
            np.tile(np.arange(1, 13), 3), start="2001-01-01", freq="MS", as_dataset=True
        )
        out = aggregate.climatological_mean(ds)
        assert out.attrs["cat:processing_level"] == "climatology"
        np.testing.assert_array_equal(out.tas, np.arange(1, 13))
