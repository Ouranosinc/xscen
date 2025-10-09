import numpy as np
import pytest
from xclim.testing.helpers import test_timeseries as timeseries

import xscen.xclim_modules.conversions as conv


def test_precipitation():
    prsn = timeseries(np.ones(3), variable="prsn", start="2001-01-01", freq="D")
    prlp = timeseries(np.ones(3) * 10, variable="pr", start="2001-01-01", freq="D", units="mm d-1")
    prlp.attrs["standard_name"] = "liquid_precipitation_flux"

    pr = conv.precipitation(prsn, prlp)
    assert pr.attrs["units"] == prsn.attrs["units"]
    np.testing.assert_array_equal(pr, prsn + prlp / 86400)


def test_tasmin_from_dtr():
    dtr = timeseries(np.ones(3) * 10, variable="tas", start="2001-01-01", freq="D", units="C")
    dtr.attrs["standard_name"] = "daily_temperature_range"
    tasmax = timeseries(np.ones(3) * 20, variable="tasmax", start="2001-01-01", freq="D")

    tasmin = conv.tasmin_from_dtr(dtr, tasmax)
    assert tasmin.attrs["units"] == tasmax.attrs["units"]
    np.testing.assert_array_equal(tasmin, tasmax - dtr)  # DTR in Celsius should not matter in the result, for a tasmax in Kelvin


def test_tasmax_from_dtr():
    dtr = timeseries(np.ones(3) * 10, variable="tas", start="2001-01-01", freq="D", units="C")
    dtr.attrs["standard_name"] = "daily_temperature_range"
    tasmin = timeseries(np.ones(3) * 10, variable="tasmin", start="2001-01-01", freq="D")

    tasmax = conv.tasmax_from_dtr(dtr, tasmin)
    assert tasmax.attrs["units"] == tasmin.attrs["units"]
    np.testing.assert_array_equal(tasmax, tasmin + dtr)  # DTR in Celsius should not matter in the result, for a tasmin in Kelvin


def test_dtr():
    tasmax = timeseries(np.ones(3) * 20, variable="tasmax", start="2001-01-01", freq="D", units="C")
    tasmin = timeseries(np.ones(3) * 10, variable="tasmin", start="2001-01-01", freq="D")

    dtr = conv.dtr_from_minmax(tasmin, tasmax)
    assert dtr.attrs["units"] == "°C"
    assert dtr.attrs["units_metadata"] == "temperature: difference"
    np.testing.assert_array_equal(dtr, (tasmax + 273.15) - tasmin)


@pytest.mark.parametrize("use_pct", (True, False))
def test_hurslogit_from_hurs(use_pct):
    factor = 100 if use_pct else 1
    hurs = timeseries(
        [0.2, 0.4, 0.6] * factor,
        variable="hurs",
        start="2001-01-01",
        freq="D",
        units="%" if use_pct else "",
    )
    hurslogit = conv.hurslogit_from_hurs(hurs)
    assert hurslogit.attrs["units"] == "1"
    np.testing.assert_array_equal(hurslogit, np.log(hurs / (factor - hurs)))


def test_hurs_from_hurslogit():
    hurs_truth = timeseries(np.array([20.0, 40.0, 60.0]), variable="hurs", start="2001-01-01", freq="D", units="%")
    # we already know that this transformation works from the test above
    hurslogit = conv.hurslogit_from_hurs(hurs_truth)
    # this tests the transformation back, and how argument xsdba_transform args are used
    hurs = conv.hurs_from_hurslogit(hurslogit)
    assert hurs.attrs["units"] == "%"
    np.testing.assert_array_almost_equal(hurs, hurs_truth)
