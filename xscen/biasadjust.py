import logging
from typing import Optional, Union

import xarray as xr
import xclim as xc
from xclim import sdba
from xclim.core.calendar import convert_calendar, get_calendar

from .catalog import parse_from_ds
from .common import minimum_calendar
from .config import parse_config

# TODO: Change all paths to PosixPath objects, including in the catalog?
# TODO: Compute sometimes fails randomly (in debug, pretty much always). Also (detrend?) fails with pr. Investigate why.

logger = logging.getLogger(__name__)


def _add_preprocessing_attr(scen, train_kwargs):
    fake_ref = xr.DataArray(name="ref")
    fake_hist = xr.DataArray(name="hist")

    preproc = []
    if train_kwargs["jitter_under"] is not None:
        preproc.append(
            xc.core.formatting.gen_call_string(
                "jitter_under_thresh", fake_ref, fake_hist, train_kwargs["jitter_under"]
            )
        )
    if train_kwargs["jitter_over"] is not None:
        preproc.append(
            xc.core.formatting.gen_call_string(
                "jitter_over_thresh", fake_ref, fake_hist, train_kwargs["jitter_over"]
            )
        )
    if train_kwargs["adapt_freq"] is not None:
        preproc.append(
            xc.core.formatting.gen_call_string(
                "adapt_freq", fake_ref, fake_hist, train_kwargs["adapt_freq"]
            )
        )

    if preproc:
        scen.attrs[
            "bias_adjustment"
        ] += ", ref and hist were prepared with " + " and ".join(preproc)


@parse_config
def train(
    dref: xr.Dataset,
    dhist: xr.Dataset,
    var: list,
    reference_period: list,
    *,
    method: str = "DetrendedQuantileMapping",
    group: Union[sdba.Grouper, str, dict] = {"group": "time.dayofyear", "window": 31},
    xclim_train_args: dict = None,
    maximal_calendar: str = "noleap",
    adapt_freq: Optional[dict] = None,
    jitter_under: Optional[dict] = None,
    jitter_over: Optional[dict] = None,
) -> xr.Dataset:
    """
    Train a bias-adjustment.

    Parameters
    ----------
    dref : xr.Dataset
      The target timeseries, on the reference period.
    dhist : xr.Dataset
      The timeseries to adjust, on the reference period.
    var : str
      Variable on which to do the adjustment
    method : str
      Name of the `sdba.TrainAdjust` method of xclim.
    group : str or sdba.Grouper
      Grouping information
    xclim_train_args : dict
      Dict of arguments to pass to the `.train` of the adjustment object.
    maximal_calendar: str
      Maximal calendar dhist can be. The hierarchy: 360_day < noleap < standard < all_leap.
      If dhist's calendar is higher than maximal calendar, it will be converted to the maximal calendar.
    reference_period : list
      [start, end] of the reference period
    adapt_freq: dict, optional
      If given, a dictionary of args to pass to the frequency adaptation function.
    jitter_under: dict, optional
      If given, a dictionary of args to pass to `jitter_under_thresh`.
    jitter_over: dict, optional
      If given, a dictionary of args to pass to `jitter_over_thresh`.


    Returns
    -------
    xr.Dataset
      Trained algorithm's data.

    """
    # TODO: To be adequately fixed later when we add multivariate
    if len(var) != 1:
        raise ValueError(
            "biasadjust currently does not support entries with multiple variables."
        )
    else:
        ref = dref[var[0]]
        hist = dhist[var[0]]

    xclim_train_args = xclim_train_args or {}
    if method == "DetrendedQuantileMapping":
        xclim_train_args.setdefault("nquantiles", 15)

    # cut out the right period
    hist = hist.sel(time=slice(*map(str, reference_period)))
    ref = ref.sel(time=slice(*map(str, reference_period)))

    # convert calendar if necessary
    simcal = get_calendar(hist)
    refcal = get_calendar(ref)
    mincal = minimum_calendar(simcal, maximal_calendar)
    if simcal != mincal:
        hist = convert_calendar(hist, mincal)
    if refcal != mincal:
        ref = convert_calendar(ref, mincal)

    if group:
        if isinstance(group, dict):
            # So we can specifiy window and add_dims in yaml.
            group = sdba.Grouper.from_kwargs(**group)["group"]
        elif isinstance(group, str):
            group = sdba.Grouper(group)
        xclim_train_args["group"] = group

    if jitter_over is not None:
        ref = sdba.processing.jitter_over_thresh(ref, **jitter_over)
        hist = sdba.processing.jitter_over_thresh(hist, **jitter_over)

    if jitter_under is not None:
        ref = sdba.processing.jitter_under_thresh(ref, **jitter_under)
        hist = sdba.processing.jitter_under_thresh(hist, **jitter_under)

    if adapt_freq is not None:
        adapt_freq.setdefault("group", group)
        hist, pth, dP0 = sdba.processing.adapt_freq(ref, hist, **adapt_freq)
        adapt_freq.pop("group")

    ADJ = getattr(sdba.adjustment, method).train(ref, hist, **xclim_train_args)

    if adapt_freq is not None:
        ds = ADJ.ds.assign(pth=pth, dP0=dP0)
    else:
        ds = ADJ.ds

    # Arguments that need to be transferred to the adjust() function
    ds.attrs["train_params"] = {
        "var": var,
        "maximal_calendar": maximal_calendar,
        "adapt_freq": adapt_freq,
        "jitter_under": jitter_under,
        "jitter_over": jitter_over,
    }

    return ds


@parse_config
def adjust(
    dtrain: xr.Dataset,
    dsim: xr.Dataset,
    simulation_period: list,
    xclim_adjust_args: dict,
    *,
    to_level: str = "biasadjusted",
    bias_adjust_institution: str = None,
    bias_adjust_project: str = None,
):
    """
    Adjust a simulation.

    Parameters
    ----------
    dtrain : xr.Dataset
      A trained algorithm's dataset, as returned by `train`.
    dsim : xr.Dataset
      Simulated timeseries, projected period.
    simulation_period : list
      [start, end] of the simulation period. or list of list if we want to adjust different period one at a time.
    xclim_adjust_args : dict
      Dict of arguments to pass to the `.adjust` of the adjustment object.
    to_level : str, optional
      The processing level to assign to the output.
      Defaults to 'biasadjusted'
    bias_adjust_institution : str, optional
      The institution to assign to the output.
    bias_adjust_project : str, optional
      The project to assign to the output.

    Returns
    -------
    xr.Dataset
      dscen, the bias-adjusted timeseries.
    """
    # TODO: To be adequately fixed later

    # evaluate the dict that was stored as a string
    if not isinstance(dtrain.attrs["train_params"], dict):
        dtrain.attrs["train_params"] = eval(dtrain.attrs["train_params"])

    var = dtrain.attrs["train_params"]["var"]
    if len(var) != 1:
        raise ValueError(
            "biasadjust currently does not support entries with multiple variables."
        )
    else:
        var = var[0]
        sim = dsim[var]

    # get right calendar
    simcal = get_calendar(sim)
    mincal = minimum_calendar(simcal, dtrain.attrs["train_params"]["maximal_calendar"])
    if simcal != mincal:
        sim = convert_calendar(sim, mincal)

    xclim_adjust_args = xclim_adjust_args or {}
    # do the adjustment for all the simulation_period lists
    if isinstance(
        simulation_period[0], str
    ):  # if only one period, turn it into a list of list
        simulation_period = [simulation_period]
    slices = []
    for period in simulation_period:
        sim = sim.sel(time=slice(period[0], period[1]))

        # adjust
        ADJ = sdba.adjustment.TrainAdjust.from_dataset(dtrain)

        if "detrend" in xclim_adjust_args and isinstance(
            xclim_adjust_args["detrend"], dict
        ):
            name, kwargs = list(xclim_adjust_args["detrend"].items())[0]
            kwargs = kwargs or {}
            kwargs.setdefault("group", ADJ.group)
            kwargs.setdefault("kind", ADJ.kind)
            xclim_adjust_args["detrend"] = getattr(sdba.detrending, name)(**kwargs)

        with xc.set_options(sdba_encode_cf=True, sdba_extra_output=False):
            out = ADJ.adjust(sim, **xclim_adjust_args)
            slices.extend([out])
    # put all the adjusted period back together
    dscen = xr.concat(slices, dim="time")

    _add_preprocessing_attr(dscen, dtrain.attrs["train_params"])
    dscen = xr.Dataset(data_vars={var: dscen}, attrs=dsim.attrs)
    # TODO: History, attrs, etc. (TODO kept from previous version of `biasadjust`)
    # TODO: Check for variables to add (grid_mapping, etc.) (TODO kept from previous version of `biasadjust`)
    dscen.attrs["cat/processing_level"] = to_level
    dscen.attrs["cat/variable"] = parse_from_ds(dscen, ["variable"])["variable"]
    if bias_adjust_institution is not None:
        dscen.attrs["cat/bias_adjust_institution"] = bias_adjust_institution
    if bias_adjust_project is not None:
        dscen.attrs["cat/bias_adjust_project"] = bias_adjust_project

    return dscen
