"""Functions to train and adjust a dataset using a bias-adjustment algorithm."""

import ast
import logging
import warnings
from copy import deepcopy

import xarray as xr
import xclim as xc
import xsdba
from xsdba.processing import from_additive_space, to_additive_space

from .catutils import parse_from_ds
from .config import parse_config
from .utils import minimum_calendar, standardize_periods, xclim_convert_units_to


logger = logging.getLogger(__name__)
xsdba.set_options(extra_output=False)


__all__ = [
    "adjust",
    "train",
]


def _add_preprocessing_attr(scen, train_kwargs):
    fake_ref = xr.DataArray(name="ref")
    fake_hist = xr.DataArray(name="hist")

    scen.attrs["bias_adjustment"] += f" with xsdba_train_args: {train_kwargs['xsdba_train_args']}"

    preproc = []

    if train_kwargs["jitter_under"] is not None:
        preproc.append(xc.core.formatting.gen_call_string("jitter_under_thresh", fake_ref, fake_hist, train_kwargs["jitter_under"]))
    if train_kwargs["jitter_over"] is not None:
        preproc.append(xc.core.formatting.gen_call_string("jitter_over_thresh", fake_ref, fake_hist, train_kwargs["jitter_over"]))

    if preproc:
        scen.attrs["bias_adjustment"] += ", ref and hist were prepared with " + " and ".join(preproc)
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
    jitter_under: dict | None = None,
    jitter_over: dict | None = None,
    align_on: str | None = "year",
    additive_space: dict | None = None,
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
      Variable(s) on which to do the adjustment.
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
    jitter_under: dict, optional
      If given, a dictionary of args to pass to `jitter_under_thresh`.
    jitter_over: dict, optional
      If given, a dictionary of args to pass to `jitter_over_thresh`.
    align_on: str, optional
      `align_on` argument for the function `xr.DataArray.convert_calendar`.
    additive_space : dict, optional
        A dictionary of variables and their arguments to convert them to additive space.
        The transformation will be applied to both `dref` and `dhist` datasets,
        as well as `dsim` and `dref` in `adjust`.
        Finally, `from_additive_space` will be called on the output of `adjust`.
        The keys are the variable names, and the values are the arguments for `to_additive_space`.
        If given, `kind` in `xsdba_train_args` must be '+'.

    Returns
    -------
    xr.Dataset
      Trained algorithm's data.

    See Also
    --------
    xsdba.adjustment.DetrendedQuantileMapping, xsdba.adjustment.ExtremeValues,
     xsdba.processing.to_additive_space, xsdba.processing.from_additive_space

    """
    if xclim_train_args is not None:
        warnings.warn(
            "`xclim_train_args` will be deprecated and replaced by `xsdba_train_args`.",
            FutureWarning,
            stacklevel=2,
        )
        if xsdba_train_args is not None:
            warnings.warn(
                "`xclim_train_args` and `xsdba_train_args` were both given, but correspond to the same option. `xsdba_train_args` will be kept",
                stacklevel=2,
            )
        else:
            xsdba_train_args = deepcopy(xclim_train_args)

    if isinstance(var, str):
        var = [var]

    # transforms
    additive_space = additive_space or {}
    if additive_space:
        for add_var, add_args in additive_space.items():
            dref[add_var] = to_additive_space(dref[add_var], **add_args)
            dhist[add_var] = to_additive_space(dhist[add_var], **add_args)
        if "kind" in xsdba_train_args and xsdba_train_args["kind"] != "+":
            warnings.warn("`additive_space` was given, but `kind` in `xsdba_train_args` is not '+'.", stacklevel=2)

    if len(var) == 1:
        ref = dref[var[0]]
        hist = dhist[var[0]]
    else:
        # Eventually, we can change ["MBCn"] and add more supported multivariate methods
        if method not in ["MBCn"]:
            raise ValueError(f"Multiple variables were given: {var}, but this treatment only works with a multivariate method,got {method}.")
        ref = xsdba.stack_variables(dref[var])
        hist = xsdba.stack_variables(dhist[var])

    group = group if group is not None else {"group": "time.dayofyear", "window": 31}
    xsdba_train_args = xsdba_train_args or {}
    xsdba_train_args_copy = deepcopy(xsdba_train_args)  # for train args
    if method == "DetrendedQuantileMapping":
        xsdba_train_args.setdefault("nquantiles", 15)

    # cut out the right period
    period = standardize_periods(period, multiple=False)
    hist = hist.sel(time=slice(*period))
    ref = ref.sel(time=slice(*period))

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

    if method != "MBCn":
        xsdba_train_args["group"] = group
    else:
        xsdba_train_args.setdefault("base_kws", {})
        xsdba_train_args["base_kws"]["group"] = group

    with xclim_convert_units_to():
        if jitter_over is not None:
            ref = xsdba.processing.jitter_over_thresh(ref, **jitter_over)
            hist = xsdba.processing.jitter_over_thresh(hist, **jitter_over)

        if jitter_under is not None:
            ref = xsdba.processing.jitter_under_thresh(ref, **jitter_under)
            hist = xsdba.processing.jitter_under_thresh(hist, **jitter_under)

        ADJ = getattr(xsdba.adjustment, method).train(ref, hist, **xsdba_train_args)

    ds = ADJ.ds

    # Arguments that need to be transferred to the adjust() function
    xsdba_train_args.pop("group", None)
    ds.attrs["train_params"] = {
        "var": var,
        "maximal_calendar": maximal_calendar,
        "xsdba_train_args": xsdba_train_args_copy,
        "jitter_under": jitter_under,
        "jitter_over": jitter_over,
        "period": period,
        "additive_space": additive_space,
    }

    # attrs that are needed to open with .to_dataset_dict()
    for a in ["cat:xrfreq", "cat:domain", "cat:id"]:
        ds.attrs[a] = dhist.attrs[a] if a in dhist.attrs else None
    ds.attrs["cat:processing_level"] = f"training_{'_'.join(var)}"

    return ds


@parse_config
def adjust(
    dtrain: xr.Dataset,
    dsim: xr.Dataset,
    periods: list[str] | list[list[str]],
    *,
    dref: xr.Dataset | None = None,
    xsdba_adjust_args: dict | None = None,
    xclim_adjust_args: dict | None = None,
    to_level: str = "biasadjusted",
    bias_adjust_institution: str | None = None,
    bias_adjust_project: str | None = None,
    bias_adjust_reference: str | None = None,
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
    dref : xr.Dataset, optional
      Reference timeseries, needed only for certain methods.
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

    Notes
    -----
    If `dref` is given as input,  `dref` and `dsim` must have the same (non-zero) number
    of time steps over training period given as input in ``xscen.train``.

    See Also
    --------
    xsdba.adjustment.DetrendedQuantileMapping, xsdba.adjustment.ExtremeValues

    """
    if xclim_adjust_args is not None:
        warnings.warn(
            "`xclim_adjust_args` will be deprecated and replaced by `xsdba_adjust_args`.",
            FutureWarning,
            stacklevel=2,
        )
        if xsdba_adjust_args is not None:
            warnings.warn(
                "`xclim_adjust_args` and `xsdba_adjust_args` were both given, but correspond to the same option. `xsdba_adjust_args` will be kept",
                stacklevel=2,
            )
        else:
            xsdba_adjust_args = deepcopy(xclim_adjust_args)
    xsdba_adjust_args = deepcopy(xsdba_adjust_args)
    xsdba_adjust_args = xsdba_adjust_args or {}

    # evaluate the dict that was stored as a string
    if not isinstance(dtrain.attrs["train_params"], dict):
        # FIXME: eval is bad. There has to be a better way!™
        dtrain.attrs["train_params"] = ast.literal_eval(dtrain.attrs["train_params"])  # noqa: S307

    # transforms
    additive_space = dtrain.attrs["train_params"]["additive_space"]
    if additive_space:
        for add_var, add_args in additive_space.items():
            if dref:
                dref[add_var] = to_additive_space(dref[add_var], **add_args)
            dsim[add_var] = to_additive_space(dsim[add_var], **add_args)

    var = dtrain.attrs["train_params"]["var"]
    if len(var) == 1:
        var = var[0]
        sim = dsim[var]
    else:
        sim = xsdba.stack_variables(dsim[var])

    # get right calendar
    simcal = sim.time.dt.calendar
    mincal = minimum_calendar(simcal, dtrain.attrs["train_params"]["maximal_calendar"])
    if simcal != mincal:
        sim = sim.convert_calendar(mincal, align_on=align_on)
    # get right calendar for `dref` too, if defined
    if dref is not None:
        ref = xsdba.stack_variables(dref[var])
        refcal = dref.time.dt.calendar
        mincal = minimum_calendar(refcal, dtrain.attrs["train_params"]["maximal_calendar"])
        if refcal != mincal:
            ref = ref.convert_calendar(mincal, align_on=align_on)

        # Used in MBCn adjusting (maybe other multivariate methods in the future)
        train_period = dtrain.attrs["train_params"]["period"]
        ref = ref.sel(time=slice(*train_period))
        hist = sim.sel(time=slice(*train_period))
        if (hist.time.size != ref.time.size) or (ref.time.size == 0):
            raise ValueError(
                " If `dref` was given as input, `dref` and `dsim` must have the same (non-zero) number of time steps "
                "over the training period `period` defined in the ``xscen.train``, but this is not the case."
            )
        xsdba_adjust_args["ref"] = ref
        xsdba_adjust_args["hist"] = hist

    # adjust
    ADJ = xsdba.adjustment.TrainAdjust.from_dataset(dtrain)

    if ("detrend" in xsdba_adjust_args) and (isinstance(xsdba_adjust_args["detrend"], dict)):
        name, kwargs = list(xsdba_adjust_args["detrend"].items())[0]
        kwargs = kwargs or {}
        kwargs.setdefault("group", ADJ.group)
        kwargs.setdefault("kind", ADJ.kind)
        xsdba_adjust_args["detrend"] = getattr(xsdba.detrending, name)(**kwargs)

    with xclim_convert_units_to():
        # do the adjustment for all the simulation_period lists
        periods = standardize_periods(periods)
        # if period_dim is specified in adjust args, use stacking
        # instead of a loop
        if period_dim := xsdba_adjust_args.get("period_dim", None):
            if len(periods) > 1:
                raise ValueError(
                    "Period stacking (`period_dim` specified in `xsdba_adjust_args`) is not allowed with multiple time slices in `periods`."
                )
            sim_stacked = xsdba.stack_periods(sim.sel(time=slice(*periods[0])), dim=period_dim)
            sim_stacked = sim_stacked.chunk({period_dim: -1})
            out = ADJ.adjust(sim_stacked, **xsdba_adjust_args)
            dscen = xsdba.unstack_periods(out)

        # do the adjustment for all the simulation_period lists
        else:
            slices = []
            for period in periods:
                sim_sel = sim.sel(time=slice(period[0], period[1]))

                out = ADJ.adjust(sim_sel, **xsdba_adjust_args)
                slices.extend([out])
            # put all the adjusted period back together
            dscen = xr.concat(slices, dim="time")

    dscen = _add_preprocessing_attr(dscen, dtrain.attrs["train_params"])
    if isinstance(var, str):
        dscen = xr.Dataset(data_vars={var: dscen}, attrs=dsim.attrs)
    else:
        dscen = xsdba.unstack_variables(dscen)

    if additive_space:
        for add_var in additive_space.keys():
            dscen[add_var] = from_additive_space(dscen[add_var])

    dscen.attrs["cat:processing_level"] = to_level
    dscen.attrs["cat:variable"] = parse_from_ds(dscen, ["variable"])["variable"]
    dscen.attrs["cat:bias_adjust_institution"] = bias_adjust_institution or "unknown"
    dscen.attrs["cat:bias_adjust_project"] = bias_adjust_project or "unknown"
    dscen.attrs["cat:bias_adjust_reference"] = bias_adjust_reference or "unknown"

    return dscen
