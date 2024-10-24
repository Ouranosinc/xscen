"""Functions to train and adjust a dataset using a bias-adjustment algorithm."""

import ast
import logging
from copy import deepcopy

import xarray as xr
import xclim as xc
from xclim import sdba

from .catutils import parse_from_ds
from .config import parse_config
from .utils import minimum_calendar, standardize_periods

logger = logging.getLogger(__name__)
xc.set_options(sdba_encode_cf=True, sdba_extra_output=False)


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


# TODO: Place this somewhere else?
def _parse_group(group):
    if isinstance(group, dict):
        # So we can specifiy window and add_dims in yaml.
        group = sdba.Grouper.from_kwargs(**group)["group"]
    elif isinstance(group, str):
        group = sdba.Grouper(group)
    return group


# TODO: Place this somewhere else?
def _harmonize_calendars(
    *inputs: list[xr.Dataset], maximal_calendar: str, align_on: str
):
    r"""Harmonize input calendars.

    Parameters
    ----------
    \*inputs : list[xr.Dataset]
      Input dataset that need calendar harmonization
    maximal_calendar :
      Maximal calendar `inputs[0]` can be. The hierarchy: 360_day < noleap < standard < all_leap.
      If `inputs[0]`'s calendar is higher than maximal calendar, it will be converted to the maximal calendar.
    align_on: str, optional
      `align_on` argument for the function `xr.Dataset.convert_calendar`.

    Returns
    -------
    list[xr.Dataset]
      Datasets with harmonized calendars
    """
    # convert to a list so that in-place changes are possible
    inputs = list(inputs)
    cals = [get_calendar(inp) for inp in inputs]
    mincal = minimum_calendar(cals[0], maximal_calendar)
    for i, cal in enumerate(cals):
        if cal != mincal:
            inputs[i] = inputs[i].convert_calendar(mincal, align_on=align_on)
    return inputs


@parse_config
def train(
    dref: xr.Dataset,
    dhist: xr.Dataset,
    var: str | list[str],
    period: list[str],
    *,
    method: str = "DetrendedQuantileMapping",
    group: sdba.Grouper | str | dict | None = None,
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
      Name of the `sdba.TrainAdjust` method of xclim.
    group : str or sdba.Grouper or dict, optional
      Grouping information. If a string, it is interpreted as a grouper on the time dimension. If a dict, it is passed to `sdba.Grouper.from_kwargs`.
      Defaults to {"group": "time.dayofyear", "window": 31}.
    xclim_train_args : dict
      Dict of arguments to pass to the `.train` of the adjustment object.
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
      `align_on` argument for the function `xr.Dataset.convert_calendar`.

    Returns
    -------
    xr.Dataset
      Trained algorithm's data.

    See Also
    --------
    xclim.sdba.adjustment.DetrendedQuantileMapping, xclim.sdba.adjustment.ExtremeValues

    """
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

    # we want to put default if group is None, but not if group is False
    if group is None:
        group = {"group": "time.dayofyear", "window": 31}

    xclim_train_args = xclim_train_args or {}
    if method == "DetrendedQuantileMapping":
        xclim_train_args.setdefault("nquantiles", 15)

    # cut out the right period
    period = standardize_periods(period, multiple=False)
    hist = hist.sel(time=slice(period[0], period[1]))
    ref = ref.sel(time=slice(period[0], period[1]))

    hist, ref = _harmonize_calendars(
        hist, ref, maximal_calendar=maximal_calendar, align_on=align_on
    )

    xclim_train_args["group"] = _parse_group(group)
    if jitter_over is not None:
        ref = sdba.processing.jitter_over_thresh(ref, **jitter_over)
        hist = sdba.processing.jitter_over_thresh(hist, **jitter_over)

    if jitter_under is not None:
        ref = sdba.processing.jitter_under_thresh(ref, **jitter_under)
        hist = sdba.processing.jitter_under_thresh(hist, **jitter_under)

    if adapt_freq is not None:
        adapt_freq.setdefault("group", xclim_train_args["group"])
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

    # attrs that are needed to open with .to_dataset_dict()
    for a in ["cat:xrfreq", "cat:domain", "cat:id"]:
        ds.attrs[a] = dhist.attrs[a] if a in dhist.attrs else None
    ds.attrs["cat:processing_level"] = f"training_{var[0]}"

    return ds


@parse_config
def adjust(
    dtrain: xr.Dataset | None,
    dsim: xr.Dataset,
    periods: list[str] | list[list[str]],
    *,
    xclim_adjust_args: dict | None = None,
    to_level: str = "biasadjusted",
    bias_adjust_institution: str | None = None,
    bias_adjust_project: str | None = None,
    align_on: str | None = "year",
    method: str | None = None,
    maximal_calendar: str = "noleap",
) -> xr.Dataset:
    """
    Adjust a simulation.

    Parameters
    ----------
    dtrain : xr.Dataset |  (optional)
      A trained algorithm's dataset, as returned by `train`. If `None`, then `method` should be provided.
    dsim : xr.Dataset
      Simulated timeseries, projected period.
    periods : list of str or list of lists of str
      Either [start, end] or list of [start, end] of the simulation periods to be adjusted (one at a time).
    xclim_adjust_args : dict, optional
      Dict of arguments to pass to the `.adjust` of the adjustment object.
    to_level : str
      The processing level to assign to the output.
      Defaults to 'biasadjusted'
    bias_adjust_institution : str, optional
      The institution to assign to the output.
    bias_adjust_project : str, optional
      The project to assign to the output.
    align_on: str, optional
      `align_on` argument for the function `xr.Dataset.convert_calendar`.
    method : str, optional
      Adjustment class. Pass this argument for a method that has no training, and thus no `dtrain` (which should be set to `None`).
      If this is set to `None`, then a non-null `dtrain` is expected.
    maximal_calendar: str, optional
      Maximal calendar dsim can be. The hierarchy: 360_day < noleap < standard < all_leap.
      If dsim's calendar is higher than maximal calendar, it will be converted to the maximal calendar.
      This is only used if `dtrain` is `None`, other the `maximal_calendar` from `dtrain` will be used.

    Returns
    -------
    xr.Dataset
      dscen, the bias-adjusted timeseries.

    See Also
    --------
    xclim.sdba.adjustment.DetrendedQuantileMapping, xclim.sdba.adjustment.ExtremeValues

    """
    if both_null := (dtrain is None and method is None) or (
        dtrain is not None and method is not None
    ):
        msg = "Both" if both_null else "Neither"
        raise ValueError(
            f"{msg} `dtrain` or `method` are `None`. One and only one of these arguments must be `None`."
        )

    xclim_adjust_args = deepcopy(xclim_adjust_args)
    xclim_adjust_args = xclim_adjust_args or {}
    if dtrain is not None:
        if not isinstance(dtrain.attrs["train_params"], dict):
            # evaluate the dict that was stored as a string
            # FIXME: eval is bad. There has to be a better way!™
            dtrain.attrs["train_params"] = ast.literal_eval(
                dtrain.attrs["train_params"]
            )

        var = dtrain.attrs["train_params"]["var"]
        if len(var) != 1:
            raise ValueError(
                "biasadjust currently does not support entries with multiple variables."
            )
        else:
            var = var[0]
            sim = dsim[var]

        (sim,) = _harmonize_calendars(
            sim,
            maximal_calendar=dtrain.attrs["train_params"]["maximal_calendar"],
            align_on=align_on,
        )
        ADJ = sdba.adjustment.TrainAdjust.from_dataset(dtrain)
    elif method is not None:
        # Adjust class assumed
        # I think in this case it would make more sense to simply have a DataArray
        # But to make it more like the rest of the method, I think it makes sense to have Dataset eith a single variable
        if len(var := list(dsim.data_vars)) > 1:
            raise ValueError(
                "When using `base`, `dsim` must contain a single variable."
            )
        var = var[0]
        # Group should be included in adjust args for Adjust-only biasadjustment
        # I think it's better to accept `xclim_adjust_args` might not be parsed appropriately,
        # anticipating the use of a YAML config file. It's OK for `sdba` not to accept a dict, and
        # it's OK for `xscen` to be aware of dict group coming from YAML files, IMO.
        # I would say that a similar (breaking) change could be made in train: Drop the explicit
        # `group` argument, just assume it will be in `xclim_train_args`.
        xclim_adjust_args.setdefault("group", {"group": "time.dayofyear", "window": 31})
        xclim_adjust_args["group"] = _parse_group(xclim_adjust_args["group"])

        sim, xclim_adjust_args["hist"], xclim_adjust_args["ref"] = _harmonize_calendars(
            dsim[var],
            xclim_adjust_args["hist"][var],
            xclim_adjust_args["ref"][var],
            maximal_calendar=maximal_calendar,
            align_on=align_on,
        )
        ADJ = getattr(sdba.adjustment, method)

    if ("detrend" in xclim_adjust_args) and (
        isinstance(xclim_adjust_args["detrend"], dict)
    ):
        name, kwargs = list(xclim_adjust_args["detrend"].items())[0]
        kwargs = kwargs or {}
        kwargs.setdefault("group", ADJ.group)
        kwargs.setdefault("kind", ADJ.kind)
        xclim_adjust_args["detrend"] = getattr(sdba.detrending, name)(**kwargs)

    # do the adjustment for all the simulation_period lists
    periods = standardize_periods(periods)
    slices = []
    for period in periods:
        sim_sel = sim.sel(time=slice(period[0], period[1]))
        out = ADJ.adjust(sim=sim_sel, **xclim_adjust_args)
        slices.extend([out])
    # put all the adjusted period back together
    dscen = xr.concat(slices, dim="time")

    if dtrain is not None:
        dscen = _add_preprocessing_attr(dscen, dtrain.attrs["train_params"])
    dscen = xr.Dataset(data_vars={var: dscen}, attrs=dsim.attrs)
    dscen.attrs["cat:processing_level"] = to_level
    dscen.attrs["cat:variable"] = parse_from_ds(dscen, ["variable"])["variable"]
    if bias_adjust_institution is not None:
        dscen.attrs["cat:bias_adjust_institution"] = bias_adjust_institution
    if bias_adjust_project is not None:
        dscen.attrs["cat:bias_adjust_project"] = bias_adjust_project

    return dscen
