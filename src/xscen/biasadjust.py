"""Functions to train and adjust a dataset using a bias-adjustment algorithm."""

import logging
import warnings
from copy import deepcopy

import xarray as xr
import xclim as xc
import xsdba
from xclim.core.units import infer_context

from .catutils import parse_from_ds
from .config import parse_config
from .utils import minimum_calendar, standardize_periods

logger = logging.getLogger(__name__)
xsdba.set_options(extra_output=False)


__all__ = [
    "adjust",
    "train",
]


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
    return scen


@parse_config
def train(
    dref: xr.Dataset,
    dhist: xr.Dataset,
    var: str | list[str],
    period: list[str],
    *,
    method: str = "DetrendedQuantileMapping",
    group: xsdba.Grouper | str | dict | None = None,
    xsdba_train_args: dict | None = None,
    xclim_train_args: dict | None = None,
    maximal_calendar: str = "noleap",
    adapt_freq: dict | None = None,
    jitter_under: dict | None = None,
    jitter_over: dict | None = None,
    align_on: str | None = "year",
) -> xr.Dataset:
    """
    Train a bias-adjustment.

    Parameters
    ----------
    dref : xr.Dataset
      The target timeseries, on the reference period.
    dhist : xr.Dataset
      The timeseries to adjust, on the reference period.
    var : str or list of str
      Variable on which to do the adjustment. Currently only supports one variable.
    period : list of str
      [start, end] of the reference period
    method : str
      Name of the `xsdba.TrainAdjust` method of xclim.
    group : str or xsdba.Grouper or dict, optional
      Grouping information. If a string, it is interpreted as a grouper on the time dimension. If a dict, it is passed to `xsdba.Grouper.from_kwargs`.
      Defaults to {"group": "time.dayofyear", "window": 31}.
    xsdba_train_args : dict, optional
      Dict of arguments to pass to the `.train` of the adjustment object.
    xclim_train_args : dict, optional
      Dict of arguments to pass to the `.train` of the adjustment object.
      A warning will be emitted stating that this a legacy argument replaced with `xsdba_train_args`.
    maximal_calendar: str
      Maximal calendar dhist can be. The hierarchy: 360_day < noleap < standard < all_leap.
      If dhist's calendar is higher than maximal calendar, it will be converted to the maximal calendar.
    adapt_freq: dict, optional
      If given, a dictionary of args to pass to the frequency adaptation function.
    jitter_under: dict, optional
      If given, a dictionary of args to pass to `jitter_under_thresh`.
    jitter_over: dict, optional
      If given, a dictionary of args to pass to `jitter_over_thresh`.
    align_on: str, optional
      `align_on` argument for the function `xr.DataArray.convert_calendar`.

    Returns
    -------
    xr.Dataset
      Trained algorithm's data.

    See Also
    --------
    xsdba.adjustment.DetrendedQuantileMapping, xsdba.adjustment.ExtremeValues

    """
    if xclim_train_args is not None:
        warnings.warn(
            "`xclim_train_args` will be deprecated and replaced by `xsdba_train_args`.",
            FutureWarning,
        )
        if xsdba_train_args is not None:
            warnings.warn(
                "`xclim_train_args` and `xsdba_train_args` were both given, but correspond to the same option. `xsdba_train_args` will be kept"
            )
        else:
            xsdba_train_args = deepcopy(xclim_train_args)
    # TODO: To be adequately fixed later when we add multivariate
    if isinstance(var, str):
        var = [var]
    if len(var) != 1:
        raise ValueError(
            "biasadjust currently does not support entries with multiple variables."
        )
    else:
        ref = dref[var[0]]
        hist = dhist[var[0]]

    group = group if group is not None else {"group": "time.dayofyear", "window": 31}

    xsdba_train_args = xsdba_train_args or {}
    if method == "DetrendedQuantileMapping":
        xsdba_train_args.setdefault("nquantiles", 15)

    # cut out the right period
    period = standardize_periods(period, multiple=False)
    hist = hist.sel(time=slice(period[0], period[1]))
    ref = ref.sel(time=slice(period[0], period[1]))

    # convert calendar if necessary
    simcal = hist.time.dt.calendar
    refcal = ref.time.dt.calendar
    mincal = minimum_calendar(simcal, maximal_calendar)
    if simcal != mincal:
        hist = hist.convert_calendar(mincal, align_on=align_on)
    if refcal != mincal:
        ref = ref.convert_calendar(mincal, align_on=align_on)

    if isinstance(group, dict):
        # So we can specify window and add_dims in yaml.
        group = xsdba.Grouper.from_kwargs(**group)["group"]
    elif isinstance(group, str):
        group = xsdba.Grouper(group)
    xsdba_train_args["group"] = group

    # TODO: change this to be compatible with multivar too?
    contexts = [
        infer_context(da.attrs.get("standard_name", None)) for da in [ref, hist]
    ]
    cntx = "hydro" if "hydro" in contexts else "none"
    with xc.core.units.units.context(cntx):
        if jitter_over is not None:
            ref = xsdba.processing.jitter_over_thresh(ref, **jitter_over)
            hist = xsdba.processing.jitter_over_thresh(hist, **jitter_over)

        if jitter_under is not None:
            ref = xsdba.processing.jitter_under_thresh(ref, **jitter_under)
            hist = xsdba.processing.jitter_under_thresh(hist, **jitter_under)

        if adapt_freq is not None:
            adapt_freq.setdefault("group", group)
            hist, pth, dP0 = xsdba.processing.adapt_freq(ref, hist, **adapt_freq)
            adapt_freq.pop("group")

        ADJ = getattr(xsdba.adjustment, method).train(ref, hist, **xsdba_train_args)

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

    # attrs that are needed to open with .to_dataset_dict()
    for a in ["cat:xrfreq", "cat:domain", "cat:id"]:
        ds.attrs[a] = dhist.attrs[a] if a in dhist.attrs else None
    ds.attrs["cat:processing_level"] = f"training_{var[0]}"

    return ds


@parse_config
def adjust(
    dtrain: xr.Dataset,
    dsim: xr.Dataset,
    periods: list[str] | list[list[str]],
    *,
    xsdba_adjust_args: dict | None = None,
    xclim_adjust_args: dict | None = None,
    to_level: str = "biasadjusted",
    bias_adjust_institution: str | None = None,
    bias_adjust_project: str | None = None,
    align_on: str | None = "year",
) -> xr.Dataset:
    """
    Adjust a simulation.

    Parameters
    ----------
    dtrain : xr.Dataset
      A trained algorithm's dataset, as returned by `train`.
    dsim : xr.Dataset
      Simulated timeseries, projected period.
    periods : list of str or list of lists of str
      Either [start, end] or list of [start, end] of the simulation periods to be adjusted (one at a time).
    xsdba_adjust_args : dict, optional
      Dict of arguments to pass to the `.adjust` of the adjustment object.
    xclim_adjust_args : dict, optional
      Dict of arguments to pass to the `.adjust` of the adjustment object
      A warning will be emitted stating that this a legacy argument replaced with `xclim_train_args`.
    to_level : str
      The processing level to assign to the output.
      Defaults to 'biasadjusted'
    bias_adjust_institution : str, optional
      The institution to assign to the output.
    bias_adjust_project : str, optional
      The project to assign to the output.
    align_on: str, optional
      `align_on` argument for the function `xr.DataArray.convert_calendar`.

    Returns
    -------
    xr.Dataset
      dscen, the bias-adjusted timeseries.

    See Also
    --------
    xsdba.adjustment.DetrendedQuantileMapping, xsdba.adjustment.ExtremeValues

    """
    if xclim_adjust_args is not None:
        warnings.warn(
            "`xclim_adjust_args` will be deprecated and replaced by `xsdba_adjust_args`.",
            FutureWarning,
        )
        if xsdba_adjust_args is not None:
            warnings.warn(
                "`xclim_adjust_args` and `xsdba_adjust_args` were both given, but correspond to the same option. `xsdba_adjust_args` will be kept"
            )
        else:
            xsdba_adjust_args = deepcopy(xclim_adjust_args)
    xsdba_adjust_args = deepcopy(xsdba_adjust_args)
    xsdba_adjust_args = xsdba_adjust_args or {}

    # evaluate the dict that was stored as a string
    if not isinstance(dtrain.attrs["train_params"], dict):
        # FIXME: eval is bad. There has to be a better way!™
        dtrain.attrs["train_params"] = eval(dtrain.attrs["train_params"])  # noqa: S307

    var = dtrain.attrs["train_params"]["var"]
    if len(var) != 1:
        raise ValueError(
            "biasadjust currently does not support entries with multiple variables."
        )
    else:
        var = var[0]
        sim = dsim[var]

    # get right calendar
    simcal = sim.time.dt.calendar
    mincal = minimum_calendar(simcal, dtrain.attrs["train_params"]["maximal_calendar"])
    if simcal != mincal:
        sim = sim.convert_calendar(mincal, align_on=align_on)

    # adjust
    ADJ = xsdba.adjustment.TrainAdjust.from_dataset(dtrain)

    if ("detrend" in xsdba_adjust_args) and (
        isinstance(xsdba_adjust_args["detrend"], dict)
    ):
        name, kwargs = list(xsdba_adjust_args["detrend"].items())[0]
        kwargs = kwargs or {}
        kwargs.setdefault("group", ADJ.group)
        kwargs.setdefault("kind", ADJ.kind)
        xsdba_adjust_args["detrend"] = getattr(xsdba.detrending, name)(**kwargs)

    cntx = infer_context(sim.attrs.get("standard_name", None))
    with xc.core.units.units.context(cntx):
        # do the adjustment for all the simulation_period lists
        periods = standardize_periods(periods)
        slices = []
        for period in periods:
            sim_sel = sim.sel(time=slice(period[0], period[1]))

            out = ADJ.adjust(sim_sel, **xsdba_adjust_args)
            slices.extend([out])
        # put all the adjusted period back together
    dscen = xr.concat(slices, dim="time")

    dscen = _add_preprocessing_attr(dscen, dtrain.attrs["train_params"])
    dscen = xr.Dataset(data_vars={var: dscen}, attrs=dsim.attrs)
    dscen.attrs["cat:processing_level"] = to_level
    dscen.attrs["cat:variable"] = parse_from_ds(dscen, ["variable"])["variable"]
    if bias_adjust_institution is not None:
        dscen.attrs["cat:bias_adjust_institution"] = bias_adjust_institution
    if bias_adjust_project is not None:
        dscen.attrs["cat:bias_adjust_project"] = bias_adjust_project

    return dscen
